"""Replot VISTA UMAPs with physical state encoded continuously.

Scenario identity is carried by marker shape. A shared logarithmic colour
scale carries mean catalogue uncertainty, which makes recovery state directly
comparable across scenarios and between the encoder and recurrent panels.
The script also audits apparent UMAP outliers in the original latent space.
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from scipy.stats import pointbiserialr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "overleaf" / "Figures"

# One physical scale across every representation figure in the paper.
UNCERTAINTY_NORM = LogNorm(vmin=0.08, vmax=30.0)
UNCERTAINTY_CMAP = plt.get_cmap("plasma")
ASSOCIATION_CMAP = LinearSegmentedColormap.from_list(
    "attention_association", ["#147DFF", "#F7F7FA", "#FF2E63"]
)
FEATURES = ["uncertainty", "age", "visibility", "quality", "slew", "relative_speed"]
FEATURE_LABELS = {
    "uncertainty": r"$U$",
    "age": "Age",
    "visibility": "Visible",
    "quality": "Quality",
    "slew": "Slew",
    "relative_speed": r"$v_{rel}$",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.3,
        "axes.labelsize": 7.6,
        "axes.titlesize": 8.5,
        "axes.titleweight": "semibold",
        "legend.fontsize": 6.8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.65,
        "axes.edgecolor": "#373241",
        "axes.labelcolor": "#373241",
        "text.color": "#373241",
        "xtick.color": "#514B5B",
        "ytick.color": "#514B5B",
        "figure.facecolor": "white",
        "axes.facecolor": "#FCFBFD",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def scenario_handles(scenarios, labels, markers):
    return [
        Line2D(
            [0], [0], linestyle="none", marker=markers[scenario], markersize=5.0,
            markerfacecolor="#817A89", markeredgecolor="white",
            markeredgewidth=0.55, label=labels[scenario],
        )
        for scenario in scenarios
    ]


def plot_umap_axis(ax, frame, x_col, y_col, scenarios, markers, title):
    for scenario in scenarios:
        cell = frame[frame["scenario"] == scenario]
        ax.scatter(
            cell[x_col], cell[y_col], c=cell["mean_u_km"],
            cmap=UNCERTAINTY_CMAP, norm=UNCERTAINTY_NORM,
            marker=markers[scenario], s=9.0, alpha=0.30,
            linewidths=0, rasterized=True,
        )

        hourly = (
            cell.assign(hour=np.rint(cell["time_h"]).clip(0, 5).astype(int))
            .groupby("hour")[[x_col, y_col, "mean_u_km"]]
            .median()
            .sort_index()
        )
        ax.plot(
            hourly[x_col], hourly[y_col], color="#625B69", linewidth=0.85,
            alpha=0.78, zorder=4,
        )
        ax.scatter(
            hourly[x_col], hourly[y_col], c=hourly["mean_u_km"],
            cmap=UNCERTAINTY_CMAP, norm=UNCERTAINTY_NORM,
            marker=markers[scenario], s=34, edgecolor="white",
            linewidth=0.65, zorder=5,
        )
        for hour in (0, 5):
            if len(scenarios) > 3:
                continue
            if hour not in hourly.index:
                continue
            point = hourly.loc[hour]
            ax.annotate(
                f"{hour} h", (point[x_col], point[y_col]), xytext=(4, 4),
                textcoords="offset points", fontsize=6.8, fontweight="semibold",
                color="#302A38",
                bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="none", alpha=0.84),
                zorder=6,
            )

    ax.set_title(title, pad=5)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)


def plot_heatmap(ax, correlations, scenarios, labels, head, features):
    matrix = np.empty((len(scenarios), len(features)))
    for row, scenario in enumerate(scenarios):
        cell = correlations[
            (correlations["scenario"] == scenario)
            & (correlations["head"] == head)
        ].set_index("feature")
        matrix[row] = cell.loc[features, "spearman_rho"].to_numpy()

    image = ax.imshow(
        matrix, cmap=ASSOCIATION_CMAP, vmin=-0.8, vmax=0.8,
        aspect="auto", interpolation="nearest",
    )
    ax.set_title(f"Head {head}", pad=4)
    ax.set_xticks(range(len(features)), [FEATURE_LABELS[item] for item in features])
    ax.tick_params(axis="x", rotation=42, pad=1)
    ax.set_yticks(range(len(scenarios)), [labels[item] for item in scenarios])
    if head != 1:
        ax.tick_params(labelleft=False)
    ax.tick_params(length=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column, row, f"{value:+.2f}", ha="center", va="center",
                fontsize=4.5,
                color="white" if abs(value) > 0.47 else "#373241",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def replace_visibility_association(correlations, raw):
    """Use point-biserial correlation for the binary visibility feature."""
    updated = correlations.copy()
    scenarios = raw["scenario"].astype(str)
    attention = raw["attention"]
    visibility = raw["token_features"][:, :, 12] > 0.5
    rows = []
    for scenario in updated["scenario"].unique():
        selected = scenarios == scenario
        for head in range(attention.shape[1]):
            scores = attention[selected, head]
            flags = visibility[selected]
            valid = np.isfinite(scores) & np.isfinite(flags)
            if np.unique(flags[valid]).size < 2:
                value = np.nan
            else:
                value = pointbiserialr(flags[valid].astype(float), scores[valid]).statistic
            rows.append((scenario, head + 1, value, int(valid.sum())))
    for scenario, head, value, count in rows:
        mask = ((updated["scenario"] == scenario)
                & (updated["head"] == head)
                & (updated["feature"] == "visibility"))
        updated.loc[mask, "spearman_rho"] = value
        updated.loc[mask, "association_method"] = "point_biserial"
        if "n_tokens" in updated:
            updated.loc[mask, "n_tokens"] = count
    updated.loc[updated["feature"] != "visibility", "association_method"] = "spearman"
    return updated


def _neighbors(values, metric, count=15):
    count = min(count + 1, len(values))
    distances, indices = NearestNeighbors(
        n_neighbors=count, metric=metric
    ).fit(values).kneighbors(values)
    return distances[:, 1:], indices[:, 1:]


def _majority(labels):
    return Counter(labels).most_common(1)[0][0]


def embedding_diagnostics(metadata, latent, embedding, representation):
    scenarios = metadata["scenario"].astype(str).to_numpy()
    latent_scaled = StandardScaler().fit_transform(latent)
    latent_distance, latent_index = _neighbors(latent_scaled, "cosine")
    projected_distance, projected_index = _neighbors(embedding, "euclidean")

    latent_cross = np.mean(scenarios[latent_index] != scenarios[:, None], axis=1)
    projected_cross = np.mean(scenarios[projected_index] != scenarios[:, None], axis=1)
    latent_majority = np.asarray([_majority(scenarios[row]) for row in latent_index])
    projected_majority = np.asarray([_majority(scenarios[row]) for row in projected_index])
    latent_score = latent_distance.mean(axis=1)
    projected_score = projected_distance.mean(axis=1)

    result = metadata.reset_index(drop=True).copy()
    result.insert(0, "representation", representation)
    result["latent_knn_distance"] = latent_score
    result["umap_knn_distance"] = projected_score
    result["latent_cross_scenario_fraction"] = latent_cross
    result["umap_cross_scenario_fraction"] = projected_cross
    result["projection_mismatch"] = projected_cross - latent_cross
    result["latent_neighbor_scenario"] = latent_majority
    result["umap_neighbor_scenario"] = projected_majority
    result["high_dimensional_outlier"] = latent_score >= np.quantile(latent_score, 0.99)
    result["projected_outlier"] = projected_score >= np.quantile(projected_score, 0.99)
    result["projection_intrusion"] = (
        (latent_majority == scenarios)
        & (projected_majority != scenarios)
        & (result["projection_mismatch"] > 0.25)
    )
    return result


def save_diagnostics(processed, metadata, raw, pre_cols, post_cols):
    pre_embedding = metadata[list(pre_cols)].to_numpy()
    post_embedding = metadata[list(post_cols)].to_numpy()
    encoder = embedding_diagnostics(metadata, raw["latent_pre"], pre_embedding, "encoder")
    recurrent = embedding_diagnostics(metadata, raw["latent_post"], post_embedding, "recurrent")
    diagnostics = pd.concat([encoder, recurrent], ignore_index=True)
    diagnostics.to_csv(processed / "umap_point_diagnostics.csv", index=False)

    audit = diagnostics[
        diagnostics["high_dimensional_outlier"] | diagnostics["projection_intrusion"]
    ].copy()
    audit["audit_priority"] = np.maximum(
        audit["latent_knn_distance"].rank(pct=True),
        audit["projection_mismatch"].rank(pct=True),
    )
    audit.sort_values("audit_priority", ascending=False).to_csv(
        processed / "umap_outlier_audit.csv", index=False
    )
    summary = (
        diagnostics.groupby(["representation", "scenario"], as_index=False)
        .agg(
            samples=("scenario", "size"),
            latent_outlier_fraction=("high_dimensional_outlier", "mean"),
            projection_intrusion_fraction=("projection_intrusion", "mean"),
            latent_scenario_mixing=("latent_cross_scenario_fraction", "mean"),
            projected_scenario_mixing=("umap_cross_scenario_fraction", "mean"),
            median_uncertainty_km=("mean_u_km", "median"),
        )
    )
    summary.to_csv(processed / "umap_diagnostic_summary.csv", index=False)
    return summary


def save_individual_panels(
    phase, latent, correlations, scenarios, labels, markers, columns,
):
    output = FIGURES / "final_review" / phase
    output.mkdir(parents=True, exist_ok=True)
    for stem, x_col, y_col, title in (
        ("umap_encoder", columns[0], columns[1], "Encoder state"),
        ("umap_recurrent", columns[2], columns[3], "Recurrent state"),
    ):
        fig, ax = plt.subplots(figsize=(3.45, 3.55))
        plot_umap_axis(ax, latent, x_col, y_col, scenarios, markers, title)
        fig.legend(
            handles=scenario_handles(scenarios, labels, markers), frameon=False,
            ncol=min(3, len(scenarios)), loc="upper center",
            bbox_to_anchor=(0.5, 1.035), columnspacing=0.7, handletextpad=0.25,
        )
        cax = fig.add_axes([0.24, 0.075, 0.56, 0.022])
        bar = fig.colorbar(
            ScalarMappable(norm=UNCERTAINTY_NORM, cmap=UNCERTAINTY_CMAP),
            cax=cax, orientation="horizontal",
        )
        bar.set_label("Mean catalogue uncertainty (km)", labelpad=1)
        bar.set_ticks([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
        bar.set_ticklabels(["0.1", "0.3", "1", "3", "10", "30"])
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.86)
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(output / f"{stem}.png", dpi=340,
                    bbox_inches="tight", pad_inches=0.025)
        plt.close(fig)

    if "visible_fraction" in latent:
        for stem, x_col, y_col, title in (
            ("umap_encoder_visible_fraction", columns[0], columns[1],
             "Encoder state: visible context"),
            ("umap_recurrent_visible_fraction", columns[2], columns[3],
             "Recurrent state: visible context"),
        ):
            fig, ax = plt.subplots(figsize=(3.45, 3.45))
            for scenario in scenarios:
                cell = latent[latent.scenario == scenario]
                ax.scatter(
                    cell[x_col], cell[y_col], c=cell.visible_fraction,
                    cmap="viridis", vmin=0, vmax=1, marker=markers[scenario],
                    s=9, alpha=0.30, linewidth=0, rasterized=True,
                )
            ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_box_aspect(1)
            ax.spines[["top", "right"]].set_visible(False)
            fig.legend(
                handles=scenario_handles(scenarios, labels, markers), frameon=False,
                ncol=min(3, len(scenarios)), loc="upper center",
                bbox_to_anchor=(0.5, 1.035), columnspacing=0.7,
            )
            cax = fig.add_axes([0.24, 0.075, 0.56, 0.022])
            bar = fig.colorbar(
                ScalarMappable(norm=Normalize(0, 1), cmap="viridis"),
                cax=cax, orientation="horizontal",
            )
            bar.set_label("Visible fraction of candidate context", labelpad=1)
            fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.86)
            fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
            fig.savefig(output / f"{stem}.png", dpi=340,
                        bbox_inches="tight", pad_inches=0.025)
            plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(7.15, 1.75))
    image = None
    for head, ax in enumerate(axes, start=1):
        image = plot_heatmap(ax, correlations, scenarios, labels, head, FEATURES)
    cax = fig.add_axes([0.35, 0.06, 0.30, 0.025])
    bar = fig.colorbar(image, cax=cax, orientation="horizontal")
    bar.set_label(r"Association ($\rho_s$; visible: $r_{pb}$)")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.91, wspace=0.30)
    fig.savefig(output / "attention_feature_associations.pdf",
                bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output / "attention_feature_associations.png", dpi=340,
                bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def render_figure(data, output_stems, scenarios, labels, markers, columns, figsize):
    processed = data / "processed"
    raw = np.load(data / "raw" / "latent_attention_samples.npz")
    latent = pd.read_csv(processed / "umap_samples.csv")
    visible = raw["token_features"][:, :, 12] > 0.5
    if "valid_k" in raw.files:
        valid_k = raw["valid_k"].astype(int)
        valid_mask = np.arange(visible.shape[1])[None, :] < valid_k[:, None]
        latent["visible_fraction"] = (
            (visible & valid_mask).sum(axis=1) / np.maximum(valid_k, 1)
        )
    else:
        latent["visible_fraction"] = visible.mean(axis=1)
    correlations = pd.read_csv(processed / "head_feature_correlations.csv")
    if "rel_speed" in correlations["feature"].unique():
        correlations["feature"] = correlations["feature"].replace(
            {"rel_speed": "relative_speed"}
        )
    correlations = replace_visibility_association(correlations, raw)
    correlations.to_csv(processed / "head_feature_associations.csv", index=False)

    summary = save_diagnostics(
        processed, latent, raw,
        (columns[0], columns[1]), (columns[2], columns[3]),
    )

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 4, height_ratios=[2.0, 1.0])
    encoder_ax = fig.add_subplot(grid[0, :2])
    recurrent_ax = fig.add_subplot(grid[0, 2:])
    plot_umap_axis(
        encoder_ax, latent, columns[0], columns[1], scenarios, markers,
        "Encoder state",
    )
    plot_umap_axis(
        recurrent_ax, latent, columns[2], columns[3], scenarios, markers,
        "Recurrent state",
    )

    heat_axes = [fig.add_subplot(grid[1, index]) for index in range(4)]
    association_image = None
    for head, ax in enumerate(heat_axes, start=1):
        association_image = plot_heatmap(
            ax, correlations, scenarios, labels, head, FEATURES
        )

    encoder_ax.text(-0.12, 1.04, "(a)", transform=encoder_ax.transAxes, fontweight="bold")
    recurrent_ax.text(-0.12, 1.04, "(b)", transform=recurrent_ax.transAxes, fontweight="bold")
    heat_axes[0].text(-0.30, 1.04, "(c)", transform=heat_axes[0].transAxes, fontweight="bold")

    fig.legend(
        handles=scenario_handles(scenarios, labels, markers), frameon=False,
        ncol=len(scenarios), loc="upper center", bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.9, handletextpad=0.35,
    )
    uncertainty_ax = fig.add_axes([0.34, 0.405, 0.32, 0.013])
    uncertainty_bar = fig.colorbar(
        ScalarMappable(norm=UNCERTAINTY_NORM, cmap=UNCERTAINTY_CMAP),
        cax=uncertainty_ax, orientation="horizontal",
    )
    uncertainty_bar.set_label("Mean catalogue uncertainty (km)", labelpad=1.5)
    uncertainty_bar.ax.xaxis.set_label_position("top")
    uncertainty_bar.set_ticks([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    uncertainty_bar.set_ticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    uncertainty_bar.outline.set_linewidth(0.45)

    association_ax = fig.add_axes([0.35, 0.055, 0.30, 0.012])
    association_bar = fig.colorbar(
        association_image, cax=association_ax, orientation="horizontal"
    )
    association_bar.set_label(
        r"Attention-feature association ($\rho_s$; visible: $r_{pb}$)"
    )
    association_bar.set_ticks([-0.8, -0.4, 0.0, 0.4, 0.8])
    association_bar.outline.set_linewidth(0.45)
    fig.subplots_adjust(
        left=0.075, right=0.99, bottom=0.14, top=0.90,
        wspace=0.32, hspace=0.52,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for stem in output_stems:
        fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
        fig.savefig(
            FIGURES / f"{stem}.png", dpi=320,
            bbox_inches="tight", pad_inches=0.025,
        )
    plt.close(fig)
    phase = "phase_I" if data.name == "attention_insights" else "phase_II"
    save_individual_panels(
        phase, latent, correlations, scenarios, labels, markers, columns,
    )
    return summary


def main():
    setup_style()
    fixed_summary = render_figure(
        ROOT / "paper" / "data" / "attention_insights",
        ("fig_phase2_attention_dynamics", "fig_attention_umap_heads", "umap",
         "phase_I_umap_attention"),
        ("id", "catalogue_120", "inclination_150"),
        {
            "id": "ID",
            "catalogue_120": "120 RSOs",
            "inclination_150": r"$i_c=150^\circ$",
        },
        {"id": "o", "catalogue_120": "s", "inclination_150": "^"},
        ("pre_umap_1", "pre_umap_2", "post_umap_1", "post_umap_2"),
        (7.25, 6.60),
    )
    large_summary = render_figure(
        ROOT / "paper" / "data" / "phase2_large_scale_attention",
        ("fig_phase2_large_scale_umap_attention", "phase_II_umap_attention"),
        (
            "id_2000_k60", "scale_5000_k60", "id_2000_k15",
            "retrograde_2000_k60", "team9_2000_k60",
        ),
        {
            "id_2000_k60": "ID: 2k, K=60",
            "scale_5000_k60": "5k RSOs",
            "id_2000_k15": "K=15",
            "retrograde_2000_k60": "Retrograde RSOs",
            "team9_2000_k60": "9 sensors",
        },
        {
            "id_2000_k60": "o", "scale_5000_k60": "s",
            "id_2000_k15": "^", "retrograde_2000_k60": "D",
            "team9_2000_k60": "P",
        },
        (
            "encoder_umap_1", "encoder_umap_2",
            "recurrent_umap_1", "recurrent_umap_2",
        ),
        (7.25, 6.60),
    )
    print("Fixed/OOD diagnostic summary:")
    print(fixed_summary.to_string(index=False))
    print("\nLarge-scale diagnostic summary:")
    print(large_summary.to_string(index=False))
    print(f"\nFigures written to {FIGURES}")


if __name__ == "__main__":
    main()
