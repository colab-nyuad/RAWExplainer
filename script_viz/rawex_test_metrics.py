'''
Knowledge Graph Embedding Metrics Visualization

Purpose:
• Load knowledge graph embedding evaluation metrics from CSV files
• Create subplots visualizing each metric (MR, MRR, Hits@1, Hits@3, Hits@10)
• Support multiple CSV file paths for comparison
• Provide modular functions for reading, plotting, and displaying results
'''

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
from datetime import datetime


def loadMetricsFromCsv(csv_path: str) -> pd.DataFrame:
    '''
    Load metrics data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing metrics
        
    Returns:
        DataFrame with metrics data
    '''
    df = pd.read_csv(csv_path)
    return df


def createMetricsSubplots(dataframes: List[pd.DataFrame], 
                          labels: List[str],
                          figsize: Tuple[int, int] = (15, 10)) -> Tuple[plt.Figure, List[plt.Axes]]:
    '''
    Create subplots for each metric in the dataframes.
    
    Args:
        dataframes: List of DataFrames containing metrics
        labels: List of labels for each dataframe (for legend)
        figsize: Figure size as (width, height)
        
    Returns:
        Tuple of (figure, list of axes)
    '''
    # Define metrics to plot
    metrics = ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@10']
    metric_labels = ['Mean Rank (MR)', 'Mean Reciprocal Rank (MRR)', 
                     'Hits@1', 'Hits@3', 'Hits@10']
    
    # Create subplots: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each metric
    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        for df, label in zip(dataframes, labels):
            # Assuming first column is the x-axis (e.g., epochs, dimensions)
            x_values = df.iloc[:, 0]
            y_values = df[metric]
            
            ax.plot(x_values, y_values, marker='o', label=label, linewidth=2)
        
        # Ensure all x-axis values are shown as ticks (no skipping)
        all_x_values = dataframes[0].iloc[:, 0]
        ax.set_xticks(all_x_values)
        ax.set_xticklabels(all_x_values)

        ax.set_xlabel('Subgraph Size (Edges, k)')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide the extra subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    return fig, axes


def plotMetricsFromPaths(csv_paths: List[str], 
                         labels: List[str] = None,
                         save_path: str = None,
                         show_plot: bool = True) -> None:
    '''
    Main function to load CSVs and plot metrics.
    
    Args:
        csv_paths: List of paths to CSV files
        labels: Optional list of labels for each CSV (defaults to filenames)
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
    '''
    # Load all dataframes
    dataframes = []
    for csv_path in csv_paths:
        df = loadMetricsFromCsv(csv_path)
        dataframes.append(df)
    
    # Generate labels if not provided
    if labels is None:
        labels = [os.path.basename(path).replace('.csv', '') for path in csv_paths]
    
    # Create plots
    fig, axes = createMetricsSubplots(dataframes, labels)
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=1900, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Show if requested
    if show_plot:
        plt.show()


if __name__ == '__main__':
    root_dir = "/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG/explanation/RAWExplainer/NBFNet/"
    dataset_name = "NELL995"
    file_name="test_all_evaluation.csv"
    csv_paths = [
        "2026-01-08-12-12-37",
        "2026-01-08-12-31-33",
        "2026-01-08-12-51-44",
        "2026-01-08-13-11-20",
        "2026-01-08-13-31-28",
        "2026-01-08-13-51-18",
        "2026-01-08-15-25-57",
        "2026-01-08-15-45-23",
        "2026-01-08-16-04-59",
        "2026-01-08-16-25-00"
    ]
    csv_paths = [os.path.join(root_dir, dataset_name, path, file_name) for path in csv_paths]

    # Optional: provide custom labels
    custom_labels = [
        str(i) for i in range(len(csv_paths))
    ]

    csv_paths = [
        "/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG/explanation/RAWExplainer/NBFNet/NELL995/2026-01-06-10-15-11-rawex-rw-re/test_all_evaluation.csv",
        "/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG/explanation/PGExplainer/NBFNet/NELL995/2026-01-06-13-30-34-both-steps/test_all_evaluation.csv"
    ]

    custom_labels = [
        "RAWExplainer",
        "PGExplainer"
    ]
    
    # SAVE PATH with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"./plots/raw_explainer_metrics/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/kg_metrics_comparison_{dataset_name}_RAWExplainer_PGExplainer.png"

    # Plot metrics
    plotMetricsFromPaths(
        csv_paths=csv_paths,
        labels=custom_labels,
        save_path=save_path,
        show_plot=True
    )