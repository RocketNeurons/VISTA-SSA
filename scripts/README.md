# Scripts

Run scripts from an editable checkout. Paths to paper data and output figures
are resolved relative to the repository, not the current working directory.
Install plotting dependencies with `pip install -e ".[plots]"`.

| Script | Purpose |
|---|---|
| `render_showcase.py` | Render the README GIF from the released VISTA checkpoints |
| `fetch_build_dependencies.py` | Fetch Raylib 5.5, verify the pinned archive SHA-256, and extract it for the native build |
| `plot_phase1_recovery_histogram.py` | Phase I recovery and uncertainty distributions |
| `plot_phase1_combined_no_colorbar.py` | Final combined Phase I figure |
| `plot_phase2_zero_shot.py` | Phase II zero-shot comparisons |
| `plot_phase2_performance_scaling_composite.py` | Phase II performance and scaling composite |
| `plot_phase3_observation_geometry_final.py` | Phase III observation geometry |
| `plot_umap_state_diagnostics.py` | UMAP state diagnostics |
| `plot_phase1_phase2_umap_attention_composite.py` | Combined Phase I/II attention and UMAP figure |

Plot scripts write into `paper/overleaf/Figures/` and may replace existing
figures. Historical CSV method identifiers are retained; displayed policy
names are VISTA and LSTM.

## Showcase animation

After installing the package and plotting dependencies, run:

```bash
python scripts/render_showcase.py
```

The animation contains Phase II steps 600–1799, showing the main uncertainty
reduction and continuing through the final 500 steps, followed by the 4000-step
Phase III episode. Both use seed 42 and stochastic actions from the released
VISTA checkpoints. The uncertainty plot records each simulation step,
including the warm-up before the captured interval.

Capture uses a hidden Raylib window and requires an OpenGL-capable display.
On a headless Linux host, install Xvfb and run:

```bash
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s "-screen 0 1920x1080x24" \
    python scripts/render_showcase.py
```

Use `--output`, `--width`, `--duration` (milliseconds per frame), and `--seed`
to customize the export. Use `--large-scale-start 1300` to show only the final
500 steps of Phase II. Endpoint PNGs and a manifest containing checkpoint
hashes, configuration, sampled steps, and uncertainty values are saved in
`runs/showcase/`; change this directory with `--work-dir`.
