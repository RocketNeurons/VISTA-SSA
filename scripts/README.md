# Scripts

Run scripts from an editable checkout. Paths to paper data and output figures
are resolved relative to the repository, not the current working directory.
Install plotting dependencies with `pip install -e ".[plots]"`.

| Script | Purpose |
|---|---|
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
