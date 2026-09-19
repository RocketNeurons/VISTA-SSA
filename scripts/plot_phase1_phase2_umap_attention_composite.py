#!/usr/bin/env python3
"""Compose Phase-I/II UMAP states and attention diagnostics in one figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_umap_state_diagnostics import (
    ASSOCIATION_CMAP,
    FEATURE_LABELS,
    FEATURES,
    UNCERTAINTY_CMAP,
    UNCERTAINTY_NORM,
    plot_umap_axis,
    scenario_handles,
    setup_style,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "overleaf" / "Figures" / "final_review"


PHASES = {
    "phase_I": {
        "data": ROOT / "paper" / "data" / "attention_insights" / "processed",
        "scenarios": ("id", "catalogue_120", "inclination_150"),
        "labels": {
            "id": r"I-ID: $N_{\rm RSO}=K=30$",
            "catalogue_120": r"I: $N_{\rm RSO}=120$",
            "inclination_150": r"I: $i_c=150^\circ$",
        },
        "attention_labels": {
            "id": "ID",
            "catalogue_120": "120 RSOs",
            "inclination_150": r"$i_c=150^\circ$",
        },
        "markers": {"id": "o", "catalogue_120": "s", "inclination_150": "^"},
        "columns": ("pre_umap_1", "pre_umap_2", "post_umap_1", "post_umap_2"),
    },
    "phase_II": {
        "data": ROOT / "paper" / "data" / "phase2_large_scale_attention" / "processed",
        "scenarios": (
            "id_2000_k60",
            "scale_5000_k60",
            "id_2000_k15",
            "retrograde_2000_k60",
            "team9_2000_k60",
        ),
        "labels": {
            "id_2000_k60": (
                r"II-ID: $N_{\rm RSO}=2{,}000,\ K=60,\ N_{\rm sens}=12$"
            ),
            "scale_5000_k60": r"II: $N_{\rm RSO}=5{,}000$",
            "id_2000_k15": r"II: $K=15$",
            "retrograde_2000_k60": "II: retrograde",
            "team9_2000_k60": r"II: $N_{\rm sens}=9$",
        },
        "attention_labels": {
            "id_2000_k60": "ID: 2k, K=60",
            "scale_5000_k60": "5k RSOs",
            "id_2000_k15": "K=15",
            "retrograde_2000_k60": "Retrograde",
            "team9_2000_k60": "9 sensors",
        },
        "markers": {
            "id_2000_k60": "D",
            "scale_5000_k60": "P",
            "id_2000_k15": "X",
            "retrograde_2000_k60": "v",
            "team9_2000_k60": "*",
        },
        "columns": (
            "encoder_umap_1",
            "encoder_umap_2",
            "recurrent_umap_1",
            "recurrent_umap_2",
        ),
    },
}


def load_phase(config):
    latent = pd.read_csv(config["data"] / "umap_samples.csv")
    associations = pd.read_csv(config["data"] / "head_feature_associations.csv")
    associations["feature"] = associations["feature"].replace(
        {"rel_speed": "relative_speed"}
    )
    return latent, associations


def plot_attention(ax, associations, config, head, show_title, show_x, show_y):
    scenarios = config["scenarios"]
    matrix = np.empty((len(scenarios), len(FEATURES)))
    for row, scenario in enumerate(scenarios):
        cell = associations[
            (associations["scenario"] == scenario)
            & (associations["head"] == head)
        ].set_index("feature")
        matrix[row] = cell.loc[FEATURES, "spearman_rho"].to_numpy()

    image = ax.imshow(
        matrix,
        cmap=ASSOCIATION_CMAP,
        vmin=-0.8,
        vmax=0.8,
        aspect="auto",
        interpolation="nearest",
    )
    if show_title:
        ax.set_title(f"Head {head}", pad=4)
    ax.set_xticks(range(len(FEATURES)))
    if show_x:
        ax.set_xticklabels([FEATURE_LABELS[item] for item in FEATURES], rotation=40)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(len(scenarios)))
    if show_y:
        row_labels = [config["attention_labels"][item] for item in scenarios]
        ax.set_yticklabels(row_labels)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=1)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=4.5,
                color="white" if abs(value) > 0.47 else "#373241",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def annotate_endpoints(ax, latent, x_col, y_col, scenarios):
    """Show trajectory direction and label the Phase-II ID endpoints."""
    for index, scenario in enumerate(scenarios):
        cell = latent[latent["scenario"] == scenario]
        hourly = (
            cell.assign(hour=np.rint(cell["time_h"]).clip(0, 5).astype(int))
            .groupby("hour")[[x_col, y_col]]
            .median()
            .sort_index()
        )
        if len(hourly) >= 2:
            start = hourly.iloc[-2]
            end = hourly.iloc[-1]
            ax.annotate(
                "",
                xy=(end[x_col], end[y_col]),
                xytext=(start[x_col], start[y_col]),
                arrowprops=dict(
                    arrowstyle="-|>", color="#4E4855", lw=0.75,
                    mutation_scale=7.5, alpha=0.9,
                ),
                zorder=6,
            )

        if index != 0:
            continue
        for hour, offset in ((0, (5, 5)), (5, (5, -10))):
            if hour not in hourly.index:
                continue
            point = hourly.loc[hour]
            ax.annotate(
                f"{hour} h",
                (point[x_col], point[y_col]),
                xytext=offset,
                textcoords="offset points",
                fontsize=5.8,
                fontweight="semibold",
                color="#302A38",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.84),
                zorder=7,
            )


def main():
    setup_style()
    plt.rcParams.update({"axes.titlesize": 8.3, "legend.fontsize": 6.5})
    loaded = {name: load_phase(config) for name, config in PHASES.items()}

    fig = plt.figure(figsize=(7.25, 10.0))
    outer = fig.add_gridspec(
        5,
        1,
        height_ratios=(1.0, 0.16, 1.0, 0.08, 0.82),
        left=0.085,
        right=0.94,
        bottom=0.070,
        top=0.970,
        hspace=0.10,
    )

    umap_axes = []
    for phase_index, phase in enumerate(("phase_I", "phase_II")):
        config = PHASES[phase]
        latent, _ = loaded[phase]
        outer_row = phase_index * 2
        grid = outer[outer_row].subgridspec(1, 2, wspace=0.11)
        row_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
        for column, ax in enumerate(row_axes):
            x_col = config["columns"][column * 2]
            y_col = config["columns"][column * 2 + 1]
            title = ("Encoder state", "Recurrent state")[column] if phase_index == 0 else ""
            plot_umap_axis(
                ax,
                latent,
                x_col,
                y_col,
                config["scenarios"],
                config["markers"],
                title,
            )
            if phase == "phase_II":
                annotate_endpoints(
                    ax, latent, x_col, y_col, config["scenarios"]
                )
            ax.set_xlabel("UMAP 1")
            if column == 0:
                ax.set_ylabel("UMAP 2")
            else:
                ax.set_ylabel("")
        umap_axes.append(row_axes)

    # One marker vocabulary and one legend cover both phases.
    legend_ax = fig.add_subplot(outer[1])
    legend_ax.axis("off")
    handles = []
    for phase in ("phase_I", "phase_II"):
        config = PHASES[phase]
        handles.extend(scenario_handles(
            config["scenarios"], config["labels"], config["markers"]
        ))
    handles.append(Line2D(
        [0], [0], color="#625B69", linewidth=0.9, marker=">",
        markersize=4.0, label=r"Hourly median: 0 h $\rightarrow$ 5 h",
    ))
    legend_ax.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="center",
        fontsize=5.7,
        columnspacing=0.75,
        labelspacing=0.35,
        handletextpad=0.25,
        borderaxespad=0.0,
    )

    phase_label_x = -0.006
    for label, row_axes in zip(("Phase I", "Phase II"), umap_axes):
        box = row_axes[0].get_position()
        fig.text(
            phase_label_x,
            0.5 * (box.y0 + box.y1),
            label,
            rotation=90,
            va="center",
            ha="center",
            weight="bold",
        )

    # The shared uncertainty scale applies to all four UMAP panels.
    upper_box = umap_axes[0][0].get_position()
    lower_box = umap_axes[1][0].get_position()
    umap_bar_ax = fig.add_axes([
        0.955,
        lower_box.y0 + 0.02,
        0.012,
        upper_box.y1 - lower_box.y0 - 0.04,
    ])
    umap_bar = fig.colorbar(
        ScalarMappable(norm=UNCERTAINTY_NORM, cmap=UNCERTAINTY_CMAP),
        cax=umap_bar_ax,
        orientation="vertical",
    )
    umap_bar.set_label("Mean catalogue uncertainty (km)", labelpad=2)
    umap_bar.set_ticks([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    umap_bar.set_ticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    umap_bar.outline.set_linewidth(0.45)

    attention = outer[4].subgridspec(2, 4, height_ratios=(3, 5), hspace=0.20, wspace=0.18)
    association_image = None
    attention_axes = []
    for row, phase in enumerate(("phase_I", "phase_II")):
        config = PHASES[phase]
        _, associations = loaded[phase]
        row_axes = []
        for column, head in enumerate(range(1, 5)):
            ax = fig.add_subplot(attention[row, column])
            row_axes.append(ax)
            association_image = plot_attention(
                ax,
                associations,
                config,
                head,
                show_title=row == 0,
                show_x=row == 1,
                show_y=column == 0,
            )
        attention_axes.append(row_axes)

    for label, row_axes in zip(("Phase I", "Phase II"), attention_axes):
        box = row_axes[0].get_position()
        fig.text(
            phase_label_x,
            0.5 * (box.y0 + box.y1),
            label,
            rotation=90,
            va="center",
            ha="center",
            weight="bold",
        )

    association_bar_ax = fig.add_axes([0.33, 0.026, 0.42, 0.012])
    association_bar = fig.colorbar(
        association_image,
        cax=association_bar_ax,
        orientation="horizontal",
    )
    association_bar.set_label(
        r"Attention-feature association ($\rho_s$; visible: $r_{pb}$)", labelpad=1
    )
    association_bar.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])
    association_bar.outline.set_linewidth(0.45)

    fig.text(0.018, upper_box.y1 + 0.012, "(a)", weight="bold")
    attention_top = attention_axes[0][0].get_position().y1
    fig.text(0.018, attention_top + 0.012, "(b)", weight="bold")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT / f"phase_I_II_umap_attention_composite.{suffix}",
            dpi=350,
            bbox_inches="tight",
            pad_inches=0.035,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
