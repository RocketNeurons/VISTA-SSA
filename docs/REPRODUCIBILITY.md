# Reproducibility

## Checkpoints

Each released checkpoint is a byte-frozen copy of a training-run artifact.
[checkpoints/manifest.json](../checkpoints/manifest.json) records, per preset:
the source run name, the original checkpoint filename, the SHA-256 of the
released file, and which config it should be evaluated with.

Verify integrity:

```bash
sha256sum checkpoints/*.pt   # compare against manifest.json
```

Every `vista eval` run also writes the SHA-256 of the checkpoint it actually
loaded into its output `manifest.json`.

## Configs

- `configs/*.ini` are the evaluation presets: they reproduce the paper's
  evaluation conditions and are what `vista eval <preset>` loads.
- `provenance/*.ini` are the original training-run configurations, kept
  verbatim for audit. For `phase1_*` and `phase2_vista` these are frozen
  copies of the exact configs the checkpoints were trained with. For
  `phase2_lstm` and `phase3_vista` they are the repository configs at release
  time; the training-run metadata was audited against the source runs listed
  in the manifest.
- `provenance/source.json` pins the PufferLib and environment source commits
  the release snapshots were taken from.

## Evaluation determinism

`vista eval` seeds NumPy and torch per episode (`seed + episode_index`), and
`--deterministic` switches from stochastic sampling to argmax action
selection. Environment dynamics are deterministic given the reset seed, so
`(preset, seed, steps, action_selection)` fully determines an episode on a
given platform. Floating-point results may still differ slightly across
CPU/GPU and hardware generations.

## Environments

`vista/envs/phase{1,2,3}` are self-contained snapshots of the three paper
environments (C sources compiled at install time against the pinned Raylib
5.5 release fetched by `scripts/fetch_build_dependencies.py`). They do not
track the upstream development repository.

## Policy implementations

| Scenario | Policy module | Policy classes | Recurrent wrappers |
|---|---|---|---|
| Phase I/II | `vista.vista_policy` | `VISTA`, `LSTM` | `VISTARecurrent`, `LSTMRecurrent` |
| Phase III | `vista.cooperative_policy` | `VISTA`, `LSTM` | `VISTARecurrent`, `LSTMRecurrent` |

The CLI selects the implementation from the phase prefix in `env_name`.
Checkpoints contain tensor state dictionaries and are loaded with strict
parameter-name and shape validation.

Source-run names in `provenance/` and checkpoint metadata identify the original
training artifacts. Paper datasets use `pointer_lstm` and `drl` for VISTA,
and `flat_lstm` or `lstm` for the LSTM baseline. Figure legends use VISTA and LSTM.
