# Phase-II Catalogue Capacity Campaign

This campaign evaluates a frozen top-K=60 Attention-LSTM pointer policy while
varying only catalogue size and the number of sensors. Each sensor-count change
preserves the three training orbit planes and distributes sensors as evenly as
possible within them.

The recovery target is mean catalogue uncertainty U <= 1 km.
A network size is considered sufficient when at least
80% of the available paired episodes reach the
target within 5 hours. Cells contain 2--6 episodes; the
exact count and confidence interval for every cell are recorded in `cells.csv`.

The reached cells give the exploratory relation T = -0.483 + 0.005350(N/S) hours (R^2 = 0.882, 13 cells).

The fitted relation is descriptive and scenario-specific. It combines physical
sensor capacity with zero-shot generalization of a policy trained using 12
sensors and 2,000 RSOs. It must not be presented as a universal SSA scaling law.

## Minimum tested sensor counts

| num_rso | minimum_tested_sensors | capacity_rso_per_sensor | criterion_met |
| --- | --- | --- | --- |
| 2000 | 9 | 222.222 | 1 |
| 3000 | 9 | 333.333 | 1 |
| 4000 | 12 | 333.333 | 1 |
| 5000 | 12 | 416.667 | 1 |
| 7500 | 18 | 416.667 | 1 |
| 10000 | 24 | 416.667 | 1 |
| 15000 | 36 | 416.667 | 1 |
| 20000 | 48 | 416.667 | 1 |

## Files

- `trajectories.csv`: mean uncertainty for every episode and sample time.
- `episodes.csv`: per-episode recovery time, censoring, and endpoint metrics.
- `cells.csv`: aggregate recovery and endpoint metrics for every N/S cell.
- `capacity.csv`: minimum tested sensor count satisfying the reliability rule.
- `campaign_config.json`: frozen checkpoint, scenario, thresholds, and grid.
