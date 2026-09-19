# Catalogue Recovery and Sensor-Network Scaling

The figure summarizes two complementary empirical scaling relationships. Only
configurations that successfully reach the recovery target are displayed.

## Recovery time

Subplot (a) shows median recovery time against catalogue load,
$N_{\mathrm{RSO}}/N_{\mathrm{sensors}}$. Recovery is defined as the first time
at which mean catalogue uncertainty falls below 1 km. The successful campaign
cells provide the empirical relation

```text
T_recovery [h] ~= -0.483 + 0.00535 (N_RSO / N_sensors),    R^2 = 0.88
```

The slope corresponds to approximately 0.321 minutes of additional recovery
time for every extra RSO assigned to each sensor, or about 32 minutes for an
increase of 100 RSOs per sensor. For example, the fit predicts recovery times
of approximately 0.59 h at 200 RSOs per sensor, 1.12 h at 300, and 1.75 h at
417. The negative intercept has no physical meaning and should not be
extrapolated toward zero load; it only positions the linear fit over the tested
range.

## Required network size

Subplot (b) reports the minimum tested number of sensors required to reduce the
mean catalogue uncertainty below 1 km within five hours. A sensor count is
considered sufficient when at least 80% of the evaluated episodes reach this
target. The policy, top-K context, orbital model, estimator, and all other
scenario parameters remain frozen while only catalogue and constellation size
change.

For the larger catalogues, the required network size grows approximately
linearly with the number of RSOs. A least-squares fit constrained through the
origin gives

```text
N_sensors ~= N_RSO / 412
```

or, equivalently, approximately 2.43 sensors per 1,000 RSOs. The directly
observed capacity boundary is close to one sensor per 417 RSOs: 12 sensors for
5,000 RSOs, 18 for 7,500, 24 for 10,000, 36 for 15,000, and 48 for 20,000.
At the largest tested scale, 42 sensors did not reach the recovery target,
whereas 48 sensors did.

The relation means that, under this sensing geometry and five-hour recovery
requirement, maintaining a roughly constant catalogue load per sensor produces
approximately linear constellation growth. It provides a useful engineering
rule for extrapolating the required sensor-network size within the tested
regime.

This is an empirical capacity law, not a universal physical scaling law. It
combines the observation capacity of the simulated constellation with the
zero-shot behaviour of a policy trained using 2,000 RSOs and 12 sensors. The
small-catalogue results also show a minimum-geometry effect: enough sensors must
be present to cover the orbital planes even when the average RSO load is low.
Consequently, the fitted ratio should be interpreted as the dominant
large-catalogue trend for this experiment, rather than as a geometry-independent
requirement for arbitrary SSA networks. The recovery-time equation is likewise
descriptive of successful cells in the sampled range and should not be used
beyond the evaluated geometry, policy, uncertainty target, or time horizon.

## Statistical qualification

The present campaign contains 4--6 episodes for most configurations up to
5,000 RSOs and 2--4 episodes for the larger catalogues. These samples establish
the preliminary trend but are not sufficient for stable 95% confidence bands,
particularly at 7,500--20,000 RSOs. The current recovery-time error bars show
the interquartile range and should not be described as confidence intervals.

For the final paper, each plotted configuration should be extended to at least
10 paired episodes and summarized using bootstrap 95% confidence intervals.
For the network-size relationship, the immediately smaller sensor count should
also be evaluated at every catalogue size. This permits the capacity boundary
to be shown as the interval between the largest unsuccessful and smallest
successful network, while bootstrap resampling can provide uncertainty bands
for the fitted scaling relationships.
