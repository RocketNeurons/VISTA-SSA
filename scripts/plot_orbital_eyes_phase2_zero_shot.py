"""Plot finite-horizon zero-shot robustness with EIG difficulty normalization."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data" / "stage12_trends_v2"
FIGURES = ROOT / "paper" / "overleaf" / "Figures"

FAMILIES = [
    "catalogue_size",
    "inclination_shift",
    "eccentricity",
    "initial_uncertainty",
    "measurement_noise",
]
METHODS = ["pointer_lstm", "flat_lstm", "expected_info_gain"]
LEARNED = ["pointer_lstm", "flat_lstm"]
LABEL = {
    "pointer_lstm": "VISTA",
    "flat_lstm": "LSTM",
    "expected_info_gain": "EIG",
}
COLOR = {
    "pointer_lstm": "#00B8D9",
    "flat_lstm": "#D642A6",
    "expected_info_gain": "#40365D",
}
MARKER = {
    "pointer_lstm": "o",
    "flat_lstm": "s",
    "expected_info_gain": "D",
}
LINESTYLE = {
    "pointer_lstm": "-",
    "flat_lstm": "--",
    "expected_info_gain": "-.",
}
TITLE = {
    "catalogue_size": "Catalogue size",
    "inclination_shift": "Inclination",
    "eccentricity": "Eccentricity",
    "initial_uncertainty": r"Initial $U$",
    "measurement_noise": "Meas. noise",
}
X_LABEL = {
    "catalogue_size": "Number of RSOs",
    "inclination_shift": "Band centre (deg)",
    "eccentricity": r"Maximum $e$",
    "initial_uncertainty": r"Initial scale ($\times$)",
    "measurement_noise": r"$\sigma_{\rm meas}$ (m)",
}
ID_X = {
    "catalogue_size": 30.0,
    "inclination_shift": 60.0,
    "eccentricity": 0.005,
    "initial_uncertainty": 1.0,
    "measurement_noise": 50.0,
}
ID_LINE_COLOR = "#8D8695"
PARITY_LINE_COLOR = "#009E7A"


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.5,
        "axes.labelsize": 7.8,
        "axes.titlesize": 8.6,
        "axes.titleweight": "semibold",
        "legend.fontsize": 7.1,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.7,
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


def display_x(family, values):
    values = np.asarray(values, dtype=float)
    if family == "inclination_shift":
        return values + 60.0
    return values


def summarize_absolute(raw):
    grouped = raw.groupby(["family", "x_value", "method"])["mean_uncertainty"]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["count"])
    return out


def difficulty_adjusted_rows(raw):
    rows = []
    for family in FAMILIES:
        family_data = raw[raw["family"] == family]
        id_value = family_data.loc[family_data["is_id"] == 1, "x_value"].iloc[0]
        for method in LEARNED:
            method_x = family_data[family_data["method"] == method][
                ["x_value", "episode_id", "mean_uncertainty"]
            ].rename(columns={"mean_uncertainty": "method_x"})
            method_id = family_data[
                (family_data["method"] == method)
                & (family_data["x_value"] == id_value)
            ][["episode_id", "mean_uncertainty"]].rename(
                columns={"mean_uncertainty": "method_id"}
            )
            eig_x = family_data[family_data["method"] == "expected_info_gain"][
                ["x_value", "episode_id", "mean_uncertainty"]
            ].rename(columns={"mean_uncertainty": "eig_x"})
            eig_id = family_data[
                (family_data["method"] == "expected_info_gain")
                & (family_data["x_value"] == id_value)
            ][["episode_id", "mean_uncertainty"]].rename(
                columns={"mean_uncertainty": "eig_id"}
            )

            paired = method_x.merge(
                method_id, on="episode_id", validate="many_to_one"
            ).merge(
                eig_x, on=["x_value", "episode_id"], validate="one_to_one"
            ).merge(
                eig_id, on="episode_id", validate="many_to_one"
            )
            paired["log_ratio"] = np.log(
                (paired["method_x"] / paired["method_id"])
                / (paired["eig_x"] / paired["eig_id"])
            )

            for x_value, cell in paired.groupby("x_value"):
                values = cell["log_ratio"].to_numpy()
                center = values.mean()
                error = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
                rows.append({
                    "family": family,
                    "x_value": float(x_value),
                    "method": method,
                    "n": int(values.size),
                    "ratio": float(np.exp(center)),
                    "ci95_low": float(np.exp(center - error)),
                    "ci95_high": float(np.exp(center + error)),
                })
    return pd.DataFrame(rows)


def legend_handles(methods):
    return [
        Line2D(
            [0], [0],
            color=COLOR[method],
            linestyle=LINESTYLE[method],
            marker=MARKER[method],
            markersize=4.2,
            markerfacecolor=COLOR[method],
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.7,
            label=LABEL[method],
        )
        for method in methods
    ]


def style_axis(ax):
    ax.grid(True, which="major")
    ax.spines[["top", "right"]].set_visible(False)


def configure_x(ax, family, bottom):
    if family == "catalogue_size":
        ticks = [30, 60, 90, 120]
    elif family == "inclination_shift":
        ticks = [30, 60, 90, 120, 150]
        ax.axvline(90, color="#B6AFBF", linewidth=0.65, linestyle=(0, (1, 2)))
        if not bottom:
            ax.text(
                91.5, 0.05, "polar",
                transform=ax.get_xaxis_transform(),
                fontsize=5.8, color="#837C8B", rotation=90, va="bottom",
            )
    elif family == "eccentricity":
        ticks = [0.005, 0.020, 0.040, 0.055]
    elif family == "initial_uncertainty":
        ticks = [0.5, 1, 2, 4]
    else:
        ticks = [25, 50, 100, 150, 200]
    ax.set_xticks(ticks)
    if family == "eccentricity":
        ax.set_xticklabels([".005", ".020", ".040", ".055"], rotation=32)
    elif family == "initial_uncertainty" and bottom:
        ax.set_xticklabels(["0.5", "1.0", "2.0", "4.0"])
        labels = ax.get_xticklabels()
        labels[0].set_ha("right")
        labels[1].set_ha("left")
    if bottom:
        ax.set_xlabel(X_LABEL[family], labelpad=4)
    else:
        ax.tick_params(labelbottom=False)


def add_id_line(ax, family, top):
    x_id = ID_X[family]
    ax.axvline(
        x_id,
        color=ID_LINE_COLOR,
        linestyle=(0, (2, 2)),
        linewidth=1.3,
        zorder=0,
    )


def finish(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if (
            box.x0 < fig.bbox.x0 - 2
            or box.y0 < fig.bbox.y0 - 2
            or box.x1 > fig.bbox.x1 + 2
            or box.y1 > fig.bbox.y1 + 2
        ):
            outside.append(item.get_text())
    if outside:
        raise RuntimeError(f"Text outside figure boundary: {outside}")

    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)


def main():
    setup_style()
    raw = pd.read_csv(DATA / "raw" / "episode_metrics.csv")
    raw = raw[raw["family"].isin(FAMILIES)].copy()
    absolute = summarize_absolute(raw)
    transfer = difficulty_adjusted_rows(raw)
    transfer.to_csv(
        DATA / "processed" / "difficulty_adjusted_transfer.csv",
        index=False,
    )

    fig, axes = plt.subplots(2, 5, figsize=(7.25, 4.75))

    for column, family in enumerate(FAMILIES):
        top = axes[0, column]
        bottom = axes[1, column]
        top.set_title(TITLE[family], pad=7)

        for method in METHODS:
            data = absolute[
                (absolute["family"] == family)
                & (absolute["method"] == method)
            ].sort_values("x_value")
            x = display_x(family, data["x_value"])
            y = data["mean"].to_numpy()
            ci = data["ci95"].to_numpy()
            top.fill_between(
                x,
                np.maximum(y - ci, 1e-3),
                y + ci,
                color=COLOR[method],
                alpha=0.105,
                linewidth=0,
            )
            top.plot(
                x,
                y,
                color=COLOR[method],
                linestyle=LINESTYLE[method],
                marker=MARKER[method],
                markersize=3.25,
                markerfacecolor=COLOR[method],
                markeredgecolor="white",
                markeredgewidth=0.4,
                linewidth=1.55,
            )

        for method in LEARNED:
            data = transfer[
                (transfer["family"] == family)
                & (transfer["method"] == method)
            ].sort_values("x_value")
            x = display_x(family, data["x_value"])
            y = data["ratio"].to_numpy()
            low = data["ci95_low"].to_numpy()
            high = data["ci95_high"].to_numpy()
            bottom.fill_between(
                x, low, high,
                color=COLOR[method],
                alpha=0.105,
                linewidth=0,
            )
            bottom.plot(
                x,
                y,
                color=COLOR[method],
                linestyle=LINESTYLE[method],
                marker=MARKER[method],
                markersize=3.25,
                markerfacecolor=COLOR[method],
                markeredgecolor="white",
                markeredgewidth=0.4,
                linewidth=1.55,
            )

        add_id_line(top, family, top=True)
        add_id_line(bottom, family, top=False)
        bottom.axhline(
            1.0,
            color=PARITY_LINE_COLOR,
            linestyle=(0, (5, 2)),
            linewidth=1.15,
            zorder=0,
        )
        style_axis(top)
        style_axis(bottom)
        configure_x(top, family, bottom=False)
        configure_x(bottom, family, bottom=True)
        bottom.xaxis.set_label_coords(0.5, -0.20)

        top.set_yscale("log")
        top.set_ylim(0.25, 80)
        top.set_yticks([0.3, 1, 3, 10, 30])
        top.set_yticklabels(["0.3", "1", "3", "10", "30"])
        top.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        bottom.set_yscale("log", base=2)
        bottom.set_ylim(0.125, 8)
        bottom.set_yticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
        bottom.set_yticklabels(["0.125", "0.25", "0.5", "1", "2", "4", "8"])
        bottom.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        if column > 0:
            top.tick_params(labelleft=False)
            bottom.tick_params(labelleft=False)

    axes[0, 0].set_ylabel(r"5-h mean $\bar U_T$ (km)")
    axes[1, 0].set_ylabel("Excess degradation vs EIG")
    reference_handles = [
        Line2D(
            [0], [0], color=ID_LINE_COLOR, linestyle="none", marker="|",
            markersize=10, markeredgewidth=1.5,
            label=r"ID setting ($N_{\rm RSO}=K=30$)",
        ),
        Line2D(
            [0], [0], color=PARITY_LINE_COLOR, linestyle=(0, (5, 2)),
            linewidth=1.3, label="Equal degradation to EIG",
        ),
    ]
    fig.legend(
        handles=legend_handles(METHODS),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        fontsize=6.4,
        handlelength=1.65,
        columnspacing=1.15,
        handletextpad=0.35,
    )
    fig.legend(
        handles=reference_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.958),
        frameon=False,
        fontsize=6.2,
        handlelength=1.55,
        columnspacing=1.25,
        handletextpad=0.35,
    )
    fig.text(
        0.012, 0.675, "Absolute outcome",
        rotation=90, va="center", ha="center",
        fontsize=6.9, fontweight="semibold", color="#6B6472",
    )
    fig.text(
        0.012, 0.300, "Zero-shot transfer",
        rotation=90, va="center", ha="center",
        fontsize=6.9, fontweight="semibold", color="#6B6472",
    )
    fig.subplots_adjust(
        left=0.092,
        right=0.985,
        bottom=0.14,
        top=0.83,
        wspace=0.24,
        hspace=0.18,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for stem in ("fig_phase2_zero_shot", "fig_stage2_ood_trends"):
        fig.savefig(FIGURES / f"{stem}.pdf")
        fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    review = FIGURES / "final_review" / "phase_I"
    review.mkdir(parents=True, exist_ok=True)
    for stem in ("Phase_I_zero_shot", "phase_I_zero_shot"):
        fig.savefig(review / f"{stem}.pdf")
        fig.savefig(review / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"[done] {FIGURES / 'fig_phase2_zero_shot.pdf'}")


if __name__ == "__main__":
    main()
