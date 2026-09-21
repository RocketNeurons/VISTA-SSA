"""Render the README showcase from the released VISTA checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from vista.config import ROOT, load, phase_of
from vista.runtime import components, make_policy


def capture(config, checkpoint, *, start, stop, stride, seed, width, work):
    cfg = load(config)
    env_type, binding = components(phase_of(cfg))
    horizon = int(cfg["env"]["max_steps"])
    if not 0 <= start < stop <= horizon:
        raise ValueError("Capture range must lie within the configured episode")
    torch.manual_seed(seed)
    env = env_type(**dict(cfg["env"], num_envs=1, render_fps=240))
    frames, metrics = [], []
    selected = set(range(start, stop, stride)) | {stop - 1}
    uncertainty = np.empty((1, cfg["env"]["num_rso"]), dtype=np.float32)
    age = np.empty_like(uncertainty)
    try:
        policy = make_policy(cfg, env, checkpoint).eval()
        obs, _ = env.reset(seed=seed)
        state = {"lstm_h": None, "lstm_c": None}
        # Initialize the renderer so it records history during uncaptured steps.
        env.render()
        with torch.no_grad():
            for step in range(stop):
                if step in selected:
                    rgb = binding.vec_render_rgb(env.c_envs, 0)
                    frame = Image.fromarray(rgb)
                    frame = frame.resize((width, round(frame.height * width / frame.width)),
                                         Image.Resampling.LANCZOS)
                    frames.append(frame)
                    binding.vec_rso_state(env.c_envs, uncertainty, age)
                    metrics.append({"step": step, "mean_u_km": float(uncertainty.mean()),
                                    "max_u_km": float(uncertainty.max())})
                    if len(frames) == 1 or step == stop - 1:
                        frame.save(work / f"{phase_of(cfg)}_{step}.png")
                if step % 100 == 0:
                    print(f"{phase_of(cfg)}: step {step}/{stop - 1}", flush=True)
                if step == stop - 1:
                    break
                logits, _ = policy.forward_eval(torch.as_tensor(obs), state)
                actions = torch.distributions.Categorical(logits=logits).sample()
                obs, *_ = env.step(actions.numpy().astype(np.int32))
    finally:
        env.close()
    return frames, {"config": cfg, "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "seed": seed, "action_selection": "sample", "start": start,
                    "stop_exclusive": stop, "stride": stride, "frames": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/vista_showcase.gif")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runs/showcase")
    parser.add_argument("--large-scale-start", type=int, default=600,
                        help="First Phase II frame; use 1300 for only the final 500 steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--duration", type=int, default=120, help="Milliseconds per GIF frame")
    args = parser.parse_args()
    if not 0 <= args.large_scale_start < 1800:
        parser.error("large-scale-start must be between 0 and 1799")
    if args.width < 1 or args.duration < 10:
        parser.error("width must be positive and duration must be at least 10 ms")
    os.environ["VISTA_RENDER_HIDDEN"] = "1"
    torch.set_num_threads(1)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_frames, clips = [], []
    for config, checkpoint, start, stop, stride in [
        (str(ROOT / "configs/large_scale.ini"), "phase2_vista", args.large_scale_start, 1800,
         max(1, (1800 - args.large_scale_start) // 168)),
        ("phase3_vista", "phase3_vista", 0, 4000, 24),
    ]:
        frames, metadata = capture(config, ROOT / f"checkpoints/{checkpoint}.pt",
                                  start=start, stop=stop, stride=stride, seed=args.seed,
                                  width=args.width, work=args.work_dir)
        all_frames.extend(frames)
        clips.append(metadata)
    # A shared palette keeps colors stable and permits compact frame differences.
    samples = all_frames[::max(1, len(all_frames) // 16)]
    sw, sh = 240, 135
    palette_image = Image.new("RGB", (sw * len(samples), sh))
    for index, frame in enumerate(samples):
        palette_image.paste(frame.resize((sw, sh)), (index * sw, 0))
    palette = palette_image.quantize(colors=256)
    all_frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE)
                  for frame in all_frames]
    temporary = args.output.with_suffix(".tmp.gif")
    all_frames[0].save(temporary, save_all=True, append_images=all_frames[1:],
                       duration=args.duration, loop=0, optimize=True, disposal=1)
    temporary.replace(args.output)
    (args.work_dir / "manifest.json").write_text(json.dumps({
        "output": str(args.output), "width": args.width,
        "duration_ms": args.duration, "clips": clips,
    }, indent=2) + "\n")
    print(f"Saved {len(all_frames)} frames to {args.output}", flush=True)


if __name__ == "__main__":
    main()
