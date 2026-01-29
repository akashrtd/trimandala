import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.figure import Figure


def load_all_results(reports_dir: str = "reports") -> Dict[str, Dict]:
    """Load all model results from reports directory"""
    results = {}

    reports_path = Path(reports_dir)
    if not reports_path.exists():
        print(f"Warning: Reports directory {reports_dir} not found")
        return results

    for model_dir in reports_path.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("."):
            model_name = model_dir.name
            results[model_name] = {}

            for track_file in model_dir.glob("*_score.json"):
                track_name = track_file.stem.replace("_score", "")
                with open(track_file, "r") as f:
                    track_data = json.load(f)

                metrics = track_data.get("metrics", {})
                results[model_name][track_name] = metrics

    return results


def plot_pareto_frontier(
    results: Dict[str, Dict],
    track: str = "TrackA",
    save_path: str = "reports/pareto_frontier.png",
    dpi: int = 300,
) -> Figure:
    """
    Plot Pareto frontier for models on a specific track.

    X-axis: Steps/Second (Log scale, higher is better)
    Y-axis: Energy Conservation Error (Log scale, inverted, lower is better)

    Dominant models (top-right) are considered state-of-the-art.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    model_data = []

    for model_name, tracks in results.items():
        if track in tracks:
            metrics = tracks[track]

            drift = metrics.get("drift", 1.0)
            latency = metrics.get("latency", 1.0)
            tes = metrics.get("tes_score", 0.0)

            # Steps per second (inverse of latency)
            steps_per_sec = 1.0 / (latency + 1e-9)

            model_data.append(
                {
                    "name": model_name,
                    "drift": drift,
                    "steps_per_sec": steps_per_sec,
                    "tes": tes,
                }
            )

    if not model_data:
        print(f"No data found for {track}")
        return fig

    # Sort by TES score for color mapping
    model_data.sort(key=lambda x: x["tes"], reverse=True)

    # Extract data
    names = [m["name"] for m in model_data]
    drifts = [m["drift"] for m in model_data]
    steps_per_sec = [m["steps_per_sec"] for m in model_data]
    tes_scores = [m["tes"] for m in model_data]

    # Normalize TES for color
    tes_norm = (np.array(tes_scores) - min(tes_scores)) / (
        max(tes_scores) - min(tes_scores) + 1e-10
    )

    # Scatter plot with color based on TES
    scatter = ax.scatter(
        steps_per_sec,
        drifts,
        c=tes_scores,
        cmap="RdYlGn",
        s=300,
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
    )

    # Add model labels
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (steps_per_sec[i], drifts[i]),
            fontsize=9,
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Identify Pareto frontier
    pareto_points = []
    for i, point in enumerate(model_data):
        is_pareto = True
        for j, other in enumerate(model_data):
            if i != j:
                # Point is dominated if:
                # Other has lower drift (better) AND higher steps/sec (better)
                if (
                    other["drift"] < point["drift"]
                    and other["steps_per_sec"] > point["steps_per_sec"]
                ):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.append(point)

    # Highlight Pareto frontier points
    if pareto_points:
        pareto_x = [p["steps_per_sec"] for p in pareto_points]
        pareto_y = [p["drift"] for p in pareto_points]
        ax.scatter(
            pareto_x,
            pareto_y,
            c="none",
            s=500,
            edgecolors="gold",
            linewidth=3,
            label="Pareto Frontier",
        )

    # Formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Inference Speed (Steps/Second)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Energy Conservation Error", fontsize=14, fontweight="bold")
    ax.set_title(
        f"Pareto Frontier: {track}\nSpeed vs Accuracy Trade-off",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("TES Score", fontsize=12, fontweight="bold")

    # Add best model annotation
    best_model = model_data[0]
    ax.text(
        0.05,
        0.95,
        f"Best: {best_model['name']}\nTES: {best_model['tes']:.2f}\n"
        f"Drift: {best_model['drift']:.2e}\nSpeed: {best_model['steps_per_sec']:.1f} steps/s",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Pareto frontier saved to: {save_path}")

    return fig


def plot_multi_track_comparison(
    results: Dict[str, Dict],
    save_path: str = "reports/multi_track_comparison.png",
    dpi: int = 300,
) -> Figure:
    """
    Compare models across all tracks in a single plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    tracks = ["TrackA", "TrackB", "TrackC"]

    for idx, track in enumerate(tracks):
        ax = axes[idx]

        model_data = []
        for model_name, tracks_data in results.items():
            if track in tracks_data:
                metrics = tracks_data[track]
                tes = metrics.get("tes_score", 0.0)
                drift = metrics.get("drift", 1.0)
                speedup = metrics.get("speedup", 1.0)

                model_data.append(
                    {
                        "name": model_name,
                        "tes": tes,
                        "drift": drift,
                        "speedup": speedup,
                    }
                )

        if model_data:
            model_data.sort(key=lambda x: x["tes"], reverse=True)

            names = [m["name"] for m in model_data]
            tes_scores = [m["tes"] for m in model_data]

            colors = [
                "green" if m["tes"] > 4.0 else "orange" if m["tes"] > 3.5 else "red"
                for m in model_data
            ]

            bars = ax.bar(
                range(len(names)),
                tes_scores,
                color=colors,
                alpha=0.7,
                edgecolor="black",
            )

            ax.set_xlabel("Model", fontsize=11, fontweight="bold")
            ax.set_ylabel("TES Score", fontsize=11, fontweight="bold")
            ax.set_title(track, fontsize=12, fontweight="bold")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, score in zip(bars, tes_scores):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.suptitle("Multi-Track Model Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Multi-track comparison saved to: {save_path}")

    return fig


def plot_data_efficiency_curve(
    data: Dict[str, List[Tuple[int, float]]],
    save_path: str = "reports/data_efficiency_comparison.png",
    dpi: int = 300,
) -> Figure:
    """
    Plot data efficiency curves comparing different models.

    Args:
        data: Dict mapping model names to list of (samples, tes_score) tuples
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name, points in data.items():
        samples = [p[0] for p in points]
        scores = [p[1] for p in points]

        ax.semilogx(
            samples, scores, marker="o", markersize=8, linewidth=2, label=model_name
        )

    ax.set_xlabel("Training Samples (Log Scale)", fontsize=14, fontweight="bold")
    ax.set_ylabel("TES Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "Data Efficiency: Learning from Few Examples", fontsize=16, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="lower right")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Data efficiency curve saved to: {save_path}")

    return fig


def generate_all_visualizations(reports_dir: str = "reports"):
    """Generate all benchmarking visualizations"""
    print("=" * 60)
    print("Generating Benchmarking Visualizations")
    print("=" * 60)

    results = load_all_results(reports_dir)

    if not results:
        print("No results found. Run benchmarks first.")
        return

    # Pareto frontiers for each track
    for track in ["TrackA", "TrackB", "TrackC"]:
        save_path = f"{reports_dir}/pareto_{track}.png"
        plot_pareto_frontier(results, track, save_path)

    # Multi-track comparison
    plot_multi_track_comparison(results, f"{reports_dir}/multi_track_comparison.png")

    print("\n✓ All visualizations generated successfully!")


if __name__ == "__main__":
    generate_all_visualizations()
