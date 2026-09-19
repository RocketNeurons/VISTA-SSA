"""Export the final Phase-I figure without the redundant ridge colorbar."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_phase1_recovery_histogram import (
    DATA,
    REVIEW,
    plot_recovery,
    plot_ridges,
    setup_style,
)


def main():
    setup_style()
    timeseries = pd.read_csv(DATA / "timeseries.csv")
    hourly = pd.read_csv(DATA / "pointer_hourly_catalog.csv")
    timeseries = timeseries[timeseries.scenario == "id_fixed_30"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95))
    plot_recovery(axes[0], timeseries, "(a)")
    plot_ridges(axes[1], hourly, "(b)")

    # Without the colorbar, panel (b)'s x-axis is the sole uncertainty scale.
    # Keep panel (a), panel spacing, and the original paper dimensions intact.
    fig.subplots_adjust(
        top=0.94,
        bottom=0.18,
        left=0.09,
        right=0.985,
        wspace=0.34,
    )

    output = REVIEW / "Phase_I_combined_no_colorbar"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(
        output.with_suffix(".png"),
        bbox_inches="tight",
        dpi=340,
        pad_inches=0.025,
    )
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
