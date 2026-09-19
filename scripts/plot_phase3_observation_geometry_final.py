#!/usr/bin/env python3
"""Render Phase-III observation geometry with the ground-site layout."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data" / "phase3_final_review"
OUTPUT = (
    ROOT / "paper" / "overleaf" / "Figures" / "final_review" / "phase_III"
)
EVENTS_FILE = DATA / "observation_events_geometry_all_directions.csv.gz"

TYPE_LABELS = {0: "Type A (near)", 1: "Type B (far)"}
SENSOR_COLORS = ("#00B8D9", "#6659E8", "#F72585", "#FF8A00")
SENSOR_MARKERS = {0: "o", 1: "^"}
SENSOR_SITES = pd.DataFrame([
    {"agent": 0, "sensor_type": 0, "latitude_deg": -4.0, "longitude_deg": -4.0},
    {"agent": 1, "sensor_type": 0, "latitude_deg": 4.0, "longitude_deg": 4.0},
    {"agent": 2, "sensor_type": 1, "latitude_deg": -4.0, "longitude_deg": 4.0},
    {"agent": 3, "sensor_type": 1, "latitude_deg": 4.0, "longitude_deg": -4.0},
])
RANGE_CROSSOVER_KM = 1900.0
EARTH_RADIUS_KM = 6371.0


def geodetic_surface_to_ecef(latitude_deg, longitude_deg):
    latitude = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    longitude = np.deg2rad(np.asarray(longitude_deg, dtype=float))
    radius = EARTH_RADIUS_KM
    return (
        radius * np.cos(latitude) * np.cos(longitude),
        radius * np.cos(latitude) * np.sin(longitude),
        radius * np.sin(latitude),
    )


def ecef_to_network_enu(x, y, z):
    """ECEF to ENU about the ground-network centre at (0 deg, 0 deg)."""
    # At lat=lon=0, east aligns with +Y, north with +Z, and up with +X.
    return (
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(x, dtype=float) - EARTH_RADIUS_KM,
    )


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.2,
        "axes.labelsize": 7.6,
        "axes.titlesize": 8.5,
        "axes.titleweight": "semibold",
        "legend.fontsize": 6.3,
        "xtick.labelsize": 6.7,
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


def render(events, layout):
    if layout == "horizontal":
        fig = plt.figure(figsize=(7.15, 3.25))
        grid = fig.add_gridspec(
            1, 2, width_ratios=(1.0, 1.14), wspace=0.30,
            left=0.065, right=0.985, bottom=0.245, top=0.875,
        )
        sky = fig.add_subplot(grid[0], projection="polar")
        geometry = fig.add_subplot(grid[1])
        sensor_legend_anchor = (0.5, 0.055)
        response_legend_anchor = (0.5, 0.012)
    else:
        fig = plt.figure(figsize=(3.55, 6.0))
        grid = fig.add_gridspec(
            2, 1, height_ratios=(1.25, 0.90), hspace=0.25,
            left=0.15, right=0.975, bottom=0.16, top=0.95,
        )
        sky = fig.add_subplot(grid[0], projection="polar")
        geometry = fig.add_subplot(grid[1])
        sensor_legend_anchor = (0.5, 0.040)
        response_legend_anchor = (0.5, 0.006)

    legend_handles = []

    # Superpose the four sensor-local sky domes. Each event retains the
    # azimuth/elevation perceived by the sensor that made the observation.
    events = events.copy()
    events["local_azimuth_rad"] = np.deg2rad(events["azimuth_deg"])
    events["local_zenith_deg"] = np.clip(90.0 - events["elevation_deg"], 0.0, 90.0)

    for sensor, group in events.groupby("agent"):
        sensor = int(sensor)
        modality = int(group["sensor_type"].iloc[0])
        color = SENSOR_COLORS[sensor % len(SENSOR_COLORS)]
        marker = SENSOR_MARKERS[modality]
        sky.scatter(
            group["local_azimuth_rad"],
            group["local_zenith_deg"],
            s=4.2, marker=marker, color=color, alpha=0.16,
            linewidth=0, rasterized=True,
        )
        geometry.scatter(
            group["slant_range_km"], group["elevation_deg"],
            s=4.2, marker=marker, color=color, alpha=0.18,
            linewidth=0, rasterized=True,
        )
        legend_handles.append(Line2D(
            [], [], color=color, marker=marker, linestyle="None",
            markersize=4.2,
            label=f"Sensor {sensor + 1} ({TYPE_LABELS[modality]})",
        ))

    site_x, site_y, site_z = geodetic_surface_to_ecef(
        SENSOR_SITES["latitude_deg"], SENSOR_SITES["longitude_deg"],
    )
    site_east, site_north, _ = ecef_to_network_enu(site_x, site_y, site_z)
    site_bearing = np.mod(np.arctan2(site_east, site_north), 2.0 * np.pi)
    # Ground-site angular offsets from the network centre are retained as an
    # inset-scale overlay near the zenith of the superposed local domes.
    site_radius = np.rad2deg(
        np.arctan2(np.hypot(site_east, site_north), site_x)
    )
    for index, row in SENSOR_SITES.iterrows():
        sensor = int(row["agent"])
        modality = int(row["sensor_type"])
        sky.scatter(
            site_bearing[index], site_radius[index], s=48,
            marker=SENSOR_MARKERS[modality], color=SENSOR_COLORS[sensor],
            edgecolor="#292530", linewidth=0.75, zorder=8,
        )
        sky.annotate(
            f"S{sensor + 1}",
            (site_bearing[index], site_radius[index]),
            xytext=(
                7.5 * np.sin(site_bearing[index]),
                7.5 * np.cos(site_bearing[index]),
            ),
            textcoords="offset points",
            ha="left" if np.sin(site_bearing[index]) >= 0 else "right",
            va="bottom" if np.cos(site_bearing[index]) >= 0 else "top",
            fontsize=5.6, fontweight="semibold", color="#292530",
            zorder=9,
        )

    sky.set_theta_zero_location("N")
    sky.set_theta_direction(-1)
    # The north tick coincides with the centered panel title; the remaining
    # compass ticks retain the azimuth reference without a typographic clash.
    sky.set_thetagrids(np.arange(45, 360, 45))
    sky.set_rlim(0, 90)
    sky.set_rticks([30, 60])
    sky.set_yticklabels([r"$60^\circ$ elev.", r"$30^\circ$ elev."])
    sky.set_rlabel_position(138)
    sky.xaxis.grid(True, color="#746A82", linewidth=0.68, alpha=0.56)
    sky.yaxis.grid(True, color="#A9A2B1", linewidth=0.55, alpha=0.65)
    sky.spines["polar"].set_linewidth(0.90)
    sky.tick_params(labelsize=6.5, pad=1)

    geometry.axvline(
        RANGE_CROSSOVER_KM, color="#68636F", linestyle=(0, (3, 2)),
        linewidth=0.95, zorder=2,
    )
    geometry.set(
        xlabel="Sensor-to-RSO range (km)",
        ylabel=r"Elevation ($^\circ$)",
    )
    geometry.grid(True)
    geometry.spines[["top", "right"]].set_visible(False)

    response_handle = Line2D(
        [], [], color="#68636F", linestyle=(0, (3, 2)), linewidth=1.0,
        label="Response crossover",
    )
    fig.legend(
        handles=legend_handles, frameon=False, ncol=2,
        loc="lower center", bbox_to_anchor=sensor_legend_anchor,
        columnspacing=0.85, handletextpad=0.35, labelspacing=0.35,
    )
    fig.legend(
        handles=[response_handle], frameon=False, ncol=1,
        loc="lower center", bbox_to_anchor=response_legend_anchor,
        handletextpad=0.40,
    )
    fig.suptitle(
        "Heterogeneous sensing geometry",
        y=0.985 if layout == "horizontal" else 0.992,
        fontsize=8.5, fontweight="semibold",
    )
    if layout == "horizontal":
        sky.text(-0.17, 1.08, "(a)", transform=sky.transAxes, fontweight="bold")
        geometry.text(-0.14, 1.08, "(b)", transform=geometry.transAxes, fontweight="bold")
    else:
        panel_x = 0.018
        fig.text(
            panel_x, sky.get_position().y1 + 0.012, "(a)",
            ha="left", va="bottom", fontweight="bold",
        )
        fig.text(
            panel_x, geometry.get_position().y1 + 0.012, "(b)",
            ha="left", va="bottom", fontweight="bold",
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    names = [f"phase_III_observation_geometry_{layout}"]
    if layout == "vertical":
        names.append("phase_III_observation_geometry")
    for name in names:
        target = OUTPUT / name
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
        fig.savefig(
            target.with_suffix(".png"), dpi=350,
            bbox_inches="tight", pad_inches=0.025,
        )
    plt.close(fig)


def main():
    setup_style()
    events = pd.read_csv(EVENTS_FILE)
    events = events[events["condition"] == "correct"].copy()
    if len(events) > 50000:
        events = events.sample(50000, random_state=42)

    # Keep the exact site layout beside the event-level data for reproducibility.
    DATA.mkdir(parents=True, exist_ok=True)
    SENSOR_SITES.to_csv(DATA / "sensor_sites.csv", index=False)
    render(events, "horizontal")
    render(events, "vertical")


if __name__ == "__main__":
    main()
