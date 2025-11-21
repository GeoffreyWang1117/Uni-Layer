"""
Visualization utilities for layer contribution analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Any
import pandas as pd


def plot_layer_contributions(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
    title: Optional[str] = None,
):
    """
    Plot layer contributions as a bar chart.

    Args:
        contributions: Output from LayerAnalyzer.compute_metrics()
        metric_name: Name of the metric to plot
        save_path: Path to save the figure
        figsize: Figure size
        title: Custom title
    """
    # Extract values
    layer_names = []
    values = []

    for layer_name, metrics in contributions.items():
        if metric_name in metrics and metrics[metric_name] is not None:
            layer_names.append(layer_name)
            values.append(metrics[metric_name])

    if not values:
        print(f"No data found for metric: {metric_name}")
        return

    # Create plot
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(values)), values, alpha=0.7, edgecolor='black')

    # Color bars by magnitude
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    plt.title(title or f"Layer Contributions: {metric_name.replace('_', ' ').title()}", fontsize=14)
    plt.xticks(range(len(layer_names)), range(len(layer_names)), rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")

    plt.show()


def plot_contribution_heatmap(
    contributions: Dict[str, Dict[str, float]],
    metrics: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    cmap: str = "RdYlGn",
):
    """
    Plot a heatmap of multiple metrics across layers.

    Args:
        contributions: Output from LayerAnalyzer.compute_metrics()
        metrics: List of metric names to include
        save_path: Path to save the figure
        figsize: Figure size
        cmap: Colormap name
    """
    # Build data matrix
    layer_names = list(contributions.keys())
    data = []

    for metric in metrics:
        row = []
        for layer_name in layer_names:
            if metric in contributions[layer_name] and contributions[layer_name][metric] is not None:
                row.append(contributions[layer_name][metric])
            else:
                row.append(np.nan)
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, index=metrics, columns=range(len(layer_names)))

    # Normalize each row
    df_normalized = df.div(df.max(axis=1), axis=0)

    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        df_normalized,
        cmap=cmap,
        annot=False,
        fmt=".2f",
        cbar_kws={'label': 'Normalized Contribution'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Metric", fontsize=12)
    plt.title("Layer Contribution Heatmap Across Metrics", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to {save_path}")

    plt.show()


def plot_depth_analysis(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str,
    num_bins: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Plot layer contributions aggregated by depth.

    Args:
        contributions: Output from LayerAnalyzer.compute_metrics()
        metric_name: Name of the metric to analyze
        num_bins: Number of depth bins
        save_path: Path to save the figure
        figsize: Figure size
    """
    num_layers = len(contributions)
    bin_size = num_layers // num_bins

    bin_labels = []
    bin_values = []

    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < num_bins - 1 else num_layers

        values = []
        for j, (layer_name, metrics) in enumerate(contributions.items()):
            if start_idx <= j < end_idx and metric_name in metrics:
                if metrics[metric_name] is not None:
                    values.append(metrics[metric_name])

        if values:
            bin_labels.append(f"Layers {start_idx}-{end_idx-1}")
            bin_values.append(np.mean(values))

    # Create plot
    plt.figure(figsize=figsize)
    plt.plot(range(len(bin_values)), bin_values, marker='o', linewidth=2, markersize=8)
    plt.fill_between(range(len(bin_values)), bin_values, alpha=0.3)

    plt.xlabel("Depth Group", fontsize=12)
    plt.ylabel(f"Average {metric_name.replace('_', ' ').title()}", fontsize=12)
    plt.title(f"Layer Contribution by Depth: {metric_name.replace('_', ' ').title()}", fontsize=14)
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=30)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved depth analysis to {save_path}")

    plt.show()


def plot_metric_comparison(
    contributions: Dict[str, Dict[str, float]],
    metric1: str,
    metric2: str,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """
    Create a scatter plot comparing two metrics across layers.

    Args:
        contributions: Output from LayerAnalyzer.compute_metrics()
        metric1: First metric name (x-axis)
        metric2: Second metric name (y-axis)
        save_path: Path to save the figure
        figsize: Figure size
    """
    x_values = []
    y_values = []
    labels = []

    for i, (layer_name, metrics) in enumerate(contributions.items()):
        if metric1 in metrics and metric2 in metrics:
            if metrics[metric1] is not None and metrics[metric2] is not None:
                x_values.append(metrics[metric1])
                y_values.append(metrics[metric2])
                labels.append(f"L{i}")

    if not x_values:
        print(f"No data found for metrics: {metric1}, {metric2}")
        return

    # Create scatter plot
    plt.figure(figsize=figsize)
    plt.scatter(x_values, y_values, s=100, alpha=0.6, c=range(len(x_values)), cmap='viridis')

    # Add labels
    for i, label in enumerate(labels):
        plt.annotate(label, (x_values[i], y_values[i]), fontsize=8, alpha=0.7)

    # Add correlation line
    if len(x_values) > 1:
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        plt.plot(x_values, p(x_values), "r--", alpha=0.5, label=f"Correlation: {np.corrcoef(x_values, y_values)[0,1]:.3f}")
        plt.legend()

    plt.xlabel(metric1.replace("_", " ").title(), fontsize=12)
    plt.ylabel(metric2.replace("_", " ").title(), fontsize=12)
    plt.title(f"Metric Comparison: {metric1} vs {metric2}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")

    plt.show()
