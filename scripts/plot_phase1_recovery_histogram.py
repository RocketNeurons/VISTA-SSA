"""Plot Phase-I recovery and hourly pointer uncertainty distributions."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data" / "phase12" / "raw"
FIGURES = ROOT / "paper" / "overleaf" / "Figures"
REVIEW = FIGURES / "final_review" / "phase_I"

ORDER = [
    "random_feasible",
    "oldest_first",
    "max_uncertainty_greedy",
    "expected_info_gain",
    "beam_a_star",
    "flat_lstm",
    "pointer_lstm",
]
LABEL = {
    "random_feasible": "Random",
    "oldest_first": "Oldest",
    "max_uncertainty_greedy": r"Max-$U$",
    "expected_info_gain": "EIG",
    "beam_a_star": r"Beam A$^*$",
    "flat_lstm": "LSTM",
    "pointer_lstm": "VISTA",
}
COLOR = {
    "random_feasible": "#9A9A9A",
    "oldest_first": "#777777",
    "max_uncertainty_greedy": "#5F5F5F",
    "expected_info_gain": "#454545",
    "beam_a_star": "#858585",
    "flat_lstm": "#555555",
    "pointer_lstm": "#111111",
}
MARKER = {
    "random_feasible": "x", "oldest_first": "v",
    "max_uncertainty_greedy": "s", "expected_info_gain": "D",
    "beam_a_star": "^", "flat_lstm": "o", "pointer_lstm": "P",
}
HOUR_COLOR = [
    "#00B8D9",
    "#2979E8",
    "#6659E8",
    "#913FD3",
    "#C32DAA",
    "#F72585",
]


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, label):
    ax.text(-0.14, 1.03, label, transform=ax.transAxes, fontweight="bold")


def log_kde(values, x_grid):
    """Return a lightly smoothed density in log10-uncertainty space."""
    samples = np.log10(np.asarray(values))
    grid = np.log10(x_grid)
    bandwidth = 0.8 * 1.06 * samples.std(ddof=1) * samples.size ** (-0.2)
    bandwidth = np.clip(bandwidth, 0.055, 0.18)
    offsets = (grid[:, None] - samples[None, :]) / bandwidth
    density = np.exp(-0.5 * offsets**2).sum(axis=1)
    density /= samples.size * bandwidth * np.sqrt(2.0 * np.pi)
    return density


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=340,
                pad_inches=0.025)
    plt.close(fig)


def plot_recovery(ax, timeseries, panel=None):
    for method in ORDER:
        d = timeseries[timeseries.method == method].sort_values("sim_time_s")
        x = d.sim_time_s.to_numpy() / 3600.0
        y = d.mean_u_km.to_numpy()
        learned = method in ("flat_lstm", "pointer_lstm")
        ax.plot(
            x, y, color=COLOR[method],
            linewidth=2.15 if method == "pointer_lstm" else (1.75 if learned else 1.1),
            linestyle="-" if learned else "--", label=LABEL[method],
            marker=MARKER[method], markevery=max(len(x) // 10, 1),
            markersize=3.5 if learned else 2.8, markerfacecolor="white",
            markeredgewidth=0.7,
            zorder=5 if method == "pointer_lstm" else (4 if learned else 2),
        )
        if learned:
            ci = 1.96 * d.std_u_km.to_numpy() / np.sqrt(d.n_episodes.to_numpy())
            ax.fill_between(x, np.maximum(y - ci, 1e-3), y + ci,
                            color=COLOR[method], alpha=0.11, linewidth=0, zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(0, 5.6)
    ax.set_xlabel("Simulated time (h)")
    ax.set_ylabel(r"Mean catalogue uncertainty $\bar U$ (km)")
    ax.grid(True, which="both", color="0.88", linewidth=0.45)
    ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=6.2,
              columnspacing=0.8, handlelength=1.7, handletextpad=0.45)
    if panel:
        panel_label(ax, panel)


def plot_ridges(ax, hourly, panel=None):
    x_grid = np.logspace(np.log10(0.02), np.log10(125.0), 420)
    cmap = plt.get_cmap("RdYlGn_r")
    norm = LogNorm(vmin=x_grid.min(), vmax=x_grid.max())
    ridge_height = 0.78
    for hour in range(6):
        values = hourly.loc[hourly.hour == hour, "u_total_km"].to_numpy()
        density = log_kde(values, x_grid)
        density = density / max(density.max(), 1e-12) * ridge_height
        baseline = float(hour)
        for index in range(len(x_grid) - 1):
            ax.fill_between(
                x_grid[index:index + 2], baseline,
                baseline + density[index:index + 2],
                color=cmap(norm(np.sqrt(x_grid[index] * x_grid[index + 1]))),
                linewidth=0, alpha=0.88,
            )
        ax.plot(x_grid, baseline + density, color="#3C3740", linewidth=0.55)
        ax.axhline(baseline, color="#BBB5BE", linewidth=0.35, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 125)
    ax.set_ylim(-0.08, 5.95)
    ax.set_yticks(range(6), [f"{hour} h" for hour in range(6)])
    ax.set_xlabel(r"Per-RSO uncertainty $U_j$ (km)")
    ax.set_ylabel("Recovery time")
    ax.spines[["top", "right"]].set_visible(False)
    if panel:
        panel_label(ax, panel)
    return ScalarMappable(norm=norm, cmap=cmap)


def main():
    setup_style()
    timeseries = pd.read_csv(DATA / "timeseries.csv")
    hourly = pd.read_csv(DATA / "pointer_hourly_catalog.csv")
    timeseries = timeseries[timeseries.scenario == "id_fixed_30"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95))
    plot_recovery(axes[0], timeseries, "(a)")
    scalar = plot_ridges(axes[1], hourly, "(b)")
    colorbar = fig.colorbar(scalar, ax=axes[1], orientation="horizontal",
                            fraction=0.045, pad=0.19, aspect=25)
    colorbar.set_label(r"Per-RSO uncertainty $U_j$ (km)", labelpad=1)

    fig.subplots_adjust(top=0.94, bottom=0.18, left=0.09, right=0.985, wspace=0.34)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for stem in ("fig_phase1_id", "fig_phase1_recovery_histogram"):
        fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    plot_recovery(ax, timeseries)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.15, top=0.98)
    save(fig, REVIEW / "phase_I_recovery_time")

    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    scalar = plot_ridges(ax, hourly)
    colorbar = fig.colorbar(scalar, ax=ax, orientation="horizontal",
                            fraction=0.045, pad=0.18, aspect=24)
    colorbar.set_label(r"Per-RSO uncertainty $U_j$ (km)", labelpad=1)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.98)
    save(fig, REVIEW / "phase_I_uncertainty_ridges")
    print(f"[done] wrote Phase-I recovery histogram figure to {FIGURES}")


if __name__ == "__main__":
    main()
