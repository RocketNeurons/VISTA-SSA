#!/usr/bin/env python3
"""Compose Phase-II performance, scaling, capacity, and compute results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
LARGE = ROOT / "paper" / "data" / "phase2_large_scale_5h_v1" / "processed"
CORRECTED_EIG = (
    ROOT / "paper" / "data" / "phase2_large_scale_5h_v1" / "corrected_eig"
)
CAPACITY = ROOT / "paper" / "data" / "phase2_catalogue_capacity"
TOPK = ROOT / "paper" / "data" / "phase2_large_scale_topk_5000"
OUTPUT = (
    ROOT / "paper" / "overleaf" / "Figures" / "final_review" / "phase_II"
    / "Phase_II_performance_scaling"
)

METHODS = ("expected_info_gain", "lstm", "drl")
LABEL = {"expected_info_gain": "EIG", "lstm": "LSTM", "drl": "VISTA"}
COLOR = {"expected_info_gain": "#4A4A4A", "lstm": "#303030", "drl": "#151515"}
LINESTYLE = {
    "expected_info_gain": (0, (5.0, 1.5, 1.2, 1.5)),
    "lstm": (0, (2.8, 1.2)),
    "drl": "-",
}
MARKER = {"expected_info_gain": "^", "lstm": "D", "drl": "o"}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "axes.titleweight": "semibold",
        "legend.fontsize": 5.8,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "axes.linewidth": 0.65,
        "axes.edgecolor": "#373241",
        "axes.labelcolor": "#373241",
        "text.color": "#373241",
        "xtick.color": "#514B5B",
        "ytick.color": "#514B5B",
        "figure.facecolor": "white",
        "axes.facecolor": "#FCFBFD",
        "grid.color": "#DDD9E3",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.72,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax, log=False):
    ax.grid(True, which="both" if log else "major")
    ax.spines[["top", "right"]].set_visible(False)


def label_panel(ax, label):
    ax.text(-0.14, 1.045, label, transform=ax.transAxes, fontweight="bold")


def method_handles():
    return [
        Line2D(
            [0], [0], color=COLOR[method], linestyle=LINESTYLE[method],
            marker=MARKER[method], markerfacecolor="white",
            markeredgewidth=0.65, markersize=3.3,
            linewidth=2.0 if method == "drl" else 1.15,
            label=LABEL[method],
        )
        for method in METHODS
    ]


def recovery_fit(cells, horizon_h=5.0):
    fit_data = cells[
        (cells["reach_fraction"] >= 0.5)
        & np.isfinite(cells["median_recovery_time_h"])
        & (cells["median_recovery_time_h"] < horizon_h)
    ]
    x = fit_data["rso_per_sensor"].to_numpy(float)
    y = fit_data["median_recovery_time_h"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    r2 = 1.0 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
    return slope, intercept, r2


def main():
    setup_style()
    timeseries = pd.read_csv(LARGE / "timeseries.csv")
    catalogue_cells = pd.read_csv(LARGE / "cells.csv")
    percentiles = pd.read_csv(LARGE / "catalog_percentiles_2000.csv")
    corrected_eig_timeseries = pd.read_csv(CORRECTED_EIG / "timeseries.csv")
    corrected_eig_summary = pd.read_csv(CORRECTED_EIG / "summary.csv")
    capacity_cells = pd.read_csv(CAPACITY / "cells.csv")
    capacity = pd.read_csv(CAPACITY / "capacity.csv")
    topk = pd.read_csv(TOPK / "topk_summary_5h.csv").sort_values("top_k")

    fig = plt.figure(figsize=(7.25, 5.55))
    outer = fig.add_gridspec(
        2, 1, height_ratios=(1.08, 0.92),
        left=0.080, right=0.955, bottom=0.105, top=0.965, hspace=0.40,
    )
    top_grid = outer[0].subgridspec(1, 2, wspace=0.32)
    bottom_grid = outer[1].subgridspec(1, 3, wspace=0.53)
    axes_top = [fig.add_subplot(top_grid[0, index]) for index in range(2)]
    axes_bottom = [fig.add_subplot(bottom_grid[0, index]) for index in range(3)]

    # (a) Reference 2,000-RSO recovery trajectory.
    ax = axes_top[0]
    reference = timeseries[timeseries["num_rso"] == 2000]
    for index, method in enumerate(METHODS):
        if method == "expected_info_gain":
            data = corrected_eig_timeseries[
                (corrected_eig_timeseries["num_rso"] == 2000)
                & (corrected_eig_timeseries["step"] <= 3600)
            ]
            data = (
                data.groupby("step", as_index=False)["mean_U"]
                .mean()
                .rename(columns={"mean_U": "meanU_mean"})
                .sort_values("step")
            )
        else:
            data = reference[reference["method"] == method].sort_values("step")
        hours = data["step"].to_numpy(float) * 5.0 / 3600.0
        mean = data["meanU_mean"].to_numpy(float)
        if method == "drl":
            band = percentiles[percentiles["method"] == method].sort_values("step")
            ax.fill_between(
                hours, band["p025_U"], band["p975_U"],
                color="#A8A8A8", alpha=0.28, linewidth=0, zorder=1,
            )
        interval = max(len(hours) // 10, 1)
        ax.plot(
            hours, mean, color=COLOR[method], linestyle=LINESTYLE[method],
            marker=MARKER[method], markevery=(2 * index, interval),
            markersize=3.2 if method == "drl" else 2.7,
            markerfacecolor="white", markeredgewidth=0.65,
            linewidth=2.05 if method == "drl" else 1.15,
            zorder=5 if method == "drl" else 3,
        )
    ax.set(
        title="Reference recovery",
        xlim=(0, 5), xlabel="Simulated time (h)",
        ylabel=r"Mean catalogue uncertainty $\bar U$ (km)",
    )
    ax.set_yscale("log")
    style_axis(ax, log=True)
    ax.legend(
        handles=method_handles() + [
            Patch(facecolor="#B8B8B8", alpha=0.45, edgecolor="none",
                  label="VISTA: central 95%")
        ],
        frameon=False, ncol=2, loc="center right", bbox_to_anchor=(0.98, 0.40),
        columnspacing=0.75, handlelength=1.7, handletextpad=0.35,
    )
    label_panel(ax, "(a)")

    # (b) Frozen-policy zero-shot catalogue-size scaling.
    ax = axes_top[1]
    for method in METHODS:
        if method == "expected_info_gain":
            data = corrected_eig_summary.sort_values("num_rso")
            x = data["num_rso"] / 1000.0
            y = data["mean_U"]
            yerr = data["mean_U_ci95"]
        else:
            data = catalogue_cells[
                catalogue_cells["method"] == method
            ].sort_values("num_rso")
            x = data["num_rso"] / 1000.0
            y = data["meanU_mean"]
            yerr = data["meanU_ci95"]
        ax.errorbar(
            x, y, yerr=yerr, color=COLOR[method],
            linestyle=LINESTYLE[method], marker=MARKER[method],
            markersize=3.9, markerfacecolor="white", markeredgewidth=0.7,
            linewidth=1.9 if method == "drl" else 1.15,
            elinewidth=0.85, capsize=2.3, capthick=0.8,
        )
    ax.set(
        title="Zero-shot catalogue scaling",
        xlabel="Catalogue size (thousand RSOs)",
        ylabel=r"Terminal mean uncertainty $\bar U_T$ (km)",
    )
    ax.set_xticks([2, 3, 4, 5])
    ax.set_yscale("log")
    style_axis(ax, log=True)
    label_panel(ax, "(b)")

    # (c) Recovery time as a function of catalogue load.
    ax = axes_bottom[0]
    successful = capacity_cells[
        (capacity_cells["reach_fraction"] >= 0.8)
        & np.isfinite(capacity_cells["median_recovery_time_h"])
    ].sort_values("rso_per_sensor")
    slope, intercept, r2 = recovery_fit(capacity_cells)
    ax.errorbar(
        successful["rso_per_sensor"], successful["median_recovery_time_h"],
        yerr=np.vstack([
            successful["median_recovery_time_h"] - successful["p25_recovery_time_h"],
            successful["p75_recovery_time_h"] - successful["median_recovery_time_h"],
        ]),
        color="#252525", linestyle="none", marker="o", markersize=3.8,
        markerfacecolor="white", markeredgewidth=0.85, capsize=1.6,
        label="Successful configurations", zorder=3,
    )
    fit_x = np.linspace(successful["rso_per_sensor"].min() * 0.94,
                        successful["rso_per_sensor"].max() * 1.04, 200)
    ax.plot(
        fit_x, intercept + slope * fit_x, color="#777777", linewidth=1.1,
        linestyle=(0, (5, 2)), label="Empirical fit",
    )
    ax.text(
        0.97, 0.025,
        rf"$T_{{rec}}={intercept:.3f}+{slope:.5f}(N_{{RSO}}/N_{{sens}})$ h"
        + "\n" + rf"$R^2={r2:.2f}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=5.1,
        color="#555555",
    )
    ax.set(
        title="Recovery-time scaling", xlabel="Catalogue load (RSOs/sensor)",
        ylabel="Median recovery time (h)",
        ylim=(0, max(2.15, successful["p75_recovery_time_h"].max() * 1.18)),
    )
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5, borderaxespad=0.2)
    label_panel(ax, "(c)")

    # (d) Minimum network size meeting the five-hour target.
    ax = axes_bottom[1]
    valid = capacity[capacity["criterion_met"] == 1].sort_values("num_rso")
    x_thousands = valid["num_rso"] / 1000.0
    ax.plot(
        x_thousands, valid["minimum_tested_sensors"], color="#222222",
        linewidth=1.3, marker="o", markersize=4.1,
        markerfacecolor="#222222", markeredgecolor="white",
        markeredgewidth=0.55, label="Minimum tested network", zorder=3,
    )
    x = valid["num_rso"].to_numpy(float)
    y = valid["minimum_tested_sensors"].to_numpy(float)
    sensor_slope = float(np.dot(x, y) / np.dot(x, x))
    fit_x = np.linspace(0, x.max() * 1.03, 200)
    ax.plot(
        fit_x / 1000.0, sensor_slope * fit_x, color="#777777",
        linewidth=1.05, linestyle=(0, (5, 2)),
        label=rf"Fit: $N_{{sens}}=N_{{RSO}}/{1 / sensor_slope:.0f}$",
    )
    ax.set(
        title="Sensor-count scaling",
        xlabel="Catalogue size (thousand RSOs)", ylabel="Minimum sensors",
    )
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left", handlelength=1.5, borderaxespad=0.2)
    label_panel(ax, "(d)")

    # (e) Candidate context accuracy-compute tradeoff.
    ax = axes_bottom[2]
    uncertainty_line = ax.errorbar(
        topk["top_k"], topk["mean_uncertainty_5h"],
        yerr=topk["ci95_5h"], color="#454148",
        linewidth=1.3, marker="o", markersize=4.2, markerfacecolor="white",
        markeredgewidth=0.95, elinewidth=0.85, capsize=2.2, capthick=0.8,
        label=r"Mean $\bar U$ (95% CI)", zorder=3,
    )
    trained = topk[topk["top_k"] == 60].iloc[0]
    training_marker = ax.scatter(
        [60], [trained["mean_uncertainty_5h"]], s=31, color="#454148",
        edgecolor="white", linewidth=0.55, zorder=4, label="Training $K$",
    )
    latency_ax = ax.twinx()
    latency_line, = latency_ax.plot(
        topk["top_k"], topk["latency_ms_step"], color="#9B969F",
        linewidth=1.15, linestyle="--", marker="s", markersize=3.8,
        markerfacecolor="white", markeredgewidth=0.85, label="Decision latency",
    )
    ax.set_xticks(topk["top_k"])
    ax.set(
        title="Context vs. compute", xlabel="Candidate-set size $K$",
        ylabel="Mean uncertainty at 5 h (km)", ylim=(0, 8),
    )
    style_axis(ax)
    latency = topk["latency_ms_step"].to_numpy(float)
    pad = 0.08 * np.ptp(latency)
    latency_ax.set_ylim(latency.min() - pad, latency.max() + pad)
    latency_ax.set_ylabel("Decision latency (ms/step)", color="#6F6973", labelpad=2)
    latency_ax.tick_params(axis="y", colors="#6F6973", pad=2)
    latency_ax.grid(False)
    latency_ax.spines["top"].set_visible(False)
    latency_ax.spines["left"].set_visible(False)
    latency_ax.spines["right"].set_color("#8F8993")
    ax.legend(
        handles=[uncertainty_line, latency_line, training_marker],
        frameon=False, ncol=2, fontsize=5.2,
        loc="upper center", bbox_to_anchor=(0.52, 0.99),
        columnspacing=0.65, handlelength=1.35, handletextpad=0.3,
        borderaxespad=0.15,
    )
    label_panel(ax, "(e)")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(
        OUTPUT.with_suffix(".png"), dpi=350,
        bbox_inches="tight", pad_inches=0.025,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
