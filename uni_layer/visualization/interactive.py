"""
Interactive visualization tools for layer contribution analysis.

This module provides enhanced visualizations including:
- Layer similarity networks
- Training dynamics over time
- Cross-model comparisons
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Any
import pandas as pd


def plot_layer_network(
    contributions: Dict[str, Dict[str, float]],
    metric_name: str = "cka_score",
    threshold: float = 0.7,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
):
    """
    Plot layer similarity as a network graph.

    Nodes are layers, edges show similarity above threshold.

    Args:
        contributions: Output from LayerAnalyzer.compute_metrics()
        metric_name: Similarity metric (e.g., 'cka_score')
        threshold: Minimum similarity to draw edge
        save_path: Path to save figure
        figsize: Figure size
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")
        return

    # Extract layer names
    layers = list(contributions.keys())
    n_layers = len(layers)

    # Create adjacency matrix (simplified - would need pairwise CKA in reality)
    # For demonstration, we'll use metric values to infer connectivity
    values = [contributions[l].get(metric_name, 0) for l in layers]

    # Create graph
    G = nx.Graph()

    # Add nodes
    for i, layer in enumerate(layers):
        G.add_node(i, name=layer, value=values[i])

    # Add edges between consecutive layers with high similarity
    for i in range(n_layers - 1):
        # Simplified: connect if both have high values
        if values[i] > threshold and values[i+1] > threshold:
            G.add_edge(i, i+1)

    # Plot
    plt.figure(figsize=figsize)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw nodes
    node_colors = [values[i] for i in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=500,
        cmap=plt.cm.viridis,
        alpha=0.9
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)

    # Draw labels
    labels = {i: f"L{i}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title(f"Layer Similarity Network ({metric_name})", fontsize=14)
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label=metric_name)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved network plot to {save_path}")

    plt.show()


def plot_metric_evolution(
    history: List[Dict[str, Dict[str, float]]],
    metric_name: str,
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
):
    """
    Plot how metrics evolve over training/time.

    Args:
        history: List of contribution dictionaries over time
        metric_name: Metric to track
        layer_names: Specific layers to plot (default: all)
        save_path: Path to save figure
        figsize: Figure size
    """
    if not history:
        print("No history data provided")
        return

    # Get all layer names from first snapshot
    all_layers = list(history[0].keys())

    if layer_names is None:
        # Select a subset for clarity
        layer_names = all_layers[::max(1, len(all_layers)//10)]  # Every 10th layer

    # Extract time series
    time_steps = list(range(len(history)))
    data = {layer: [] for layer in layer_names}

    for snapshot in history:
        for layer in layer_names:
            value = snapshot.get(layer, {}).get(metric_name, np.nan)
            data[layer].append(value)

    # Plot
    plt.figure(figsize=figsize)

    for layer in layer_names:
        plt.plot(time_steps, data[layer], marker='o', label=layer, alpha=0.7)

    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    plt.title(f"Metric Evolution: {metric_name}", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved evolution plot to {save_path}")

    plt.show()


def plot_cross_model_comparison(
    models_contributions: Dict[str, Dict[str, Dict[str, float]]],
    metric_name: str,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    """
    Compare layer contributions across multiple models.

    Args:
        models_contributions: {model_name: contributions_dict}
        metric_name: Metric to compare
        save_path: Path to save figure
        figsize: Figure size
    """
    # Prepare data for plotting
    data = []

    for model_name, contributions in models_contributions.items():
        for layer_idx, (layer_name, metrics) in enumerate(contributions.items()):
            value = metrics.get(metric_name)
            if value is not None:
                data.append({
                    'Model': model_name,
                    'Layer Index': layer_idx,
                    'Value': value
                })

    if not data:
        print("No data to plot")
        return

    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=figsize)

    # Line plot for each model
    for model_name in df['Model'].unique():
        model_data = df[df['Model'] == model_name]
        plt.plot(
            model_data['Layer Index'],
            model_data['Value'],
            marker='o',
            label=model_name,
            linewidth=2,
            markersize=6
        )

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    plt.title(f"Cross-Model Comparison: {metric_name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")

    plt.show()


def plot_radar_chart(
    contributions: Dict[str, Dict[str, float]],
    layer_name: str,
    metrics: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
):
    """
    Create radar chart showing multiple metrics for a single layer.

    Args:
        contributions: Output from LayerAnalyzer.compute_metrics()
        layer_name: Name of layer to visualize
        metrics: List of metric names to include
        save_path: Path to save figure
        figsize: Figure size
    """
    if layer_name not in contributions:
        print(f"Layer '{layer_name}' not found")
        return

    layer_metrics = contributions[layer_name]

    # Extract values (normalize to [0, 1])
    values = []
    labels = []

    for metric in metrics:
        if metric in layer_metrics and layer_metrics[metric] is not None:
            values.append(layer_metrics[metric])
            labels.append(metric.replace("_", " ").title())

    if not values:
        print("No valid metrics found")
        return

    # Normalize values
    values = np.array(values)
    values_normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)

    # Number of variables
    N = len(labels)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_normalized = np.concatenate((values_normalized, [values_normalized[0]]))
    angles += angles[:1]

    # Plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    ax.plot(angles, values_normalized, 'o-', linewidth=2, label=layer_name)
    ax.fill(angles, values_normalized, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Layer Profile: {layer_name}", size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved radar chart to {save_path}")

    plt.show()
