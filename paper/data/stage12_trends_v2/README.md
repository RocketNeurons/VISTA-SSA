# Phase-II zero-shot scalar sweeps

This package contains the finite-horizon transfer data used by
`fig_phase2_zero_shot`. The frozen attention-LSTM pointer policy, flat LSTM,
and expected-information-gain (EIG) scheduler are evaluated for 5 h with
matched seeds. No policy is retrained or fine-tuned in a shifted scenario.

## Figure metrics

The upper row reports absolute final mean catalogue uncertainty with 95%
confidence intervals. The lower row reports difficulty-adjusted transfer:

`[(U_method(x) / U_method(ID)) / (U_EIG(x) / U_EIG(ID))]`.

A value of one means that the learned method degrades at the same rate as EIG.
Values below one indicate better relative transfer, while values above one
indicate excess degradation beyond the shift's effect on the classical
scheduler. This is a finite-horizon zero-shot diagnostic, not an estimate of
the stationary performance attainable after retraining.

## Files

- `raw/episode_metrics.csv`: one row per sweep, point, method, and paired
  episode.
- `processed/summary.csv`: means and 95% confidence intervals for each cell.
- `processed/difficulty_adjusted_transfer.csv`: figure-ready paired
  difficulty-adjusted ratios.
- `metadata.json`: complete point definitions, environment overrides, seeds,
  and execution settings.

Each cell contains 24 paired episodes. Existing points were imported from
`paper/data/stage12_trends`; the evaluator ran only missing inclination,
eccentricity, initial-uncertainty, and measurement-noise points.
