import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def read_single_mrr_file(csv_file_path):
    """
    Read MRR data from a single CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing MRR data
        
    Returns:
        tuple: (edge_drop_probabilities, mrr_values)
            - edge_drop_probabilities (list): List of edge-drop probability values
            - mrr_values (list): MRR values from the second row
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path, index_col=0)
    
    # Extract edge-drop probabilities from column headers
    edge_drop_probabilities = [float(col) for col in df.columns]
    
    # Extract MRR values from the first data row (second row in file)
    first_row = df.iloc[1]
    mrr_values = first_row.tolist()
    
    return edge_drop_probabilities, mrr_values


def read_mrr_data_from_files(phi_file_path, phi_eval_file_path):
    """
    Read MRR data from two separate CSV files.
    
    Args:
        phi_file_path (str): Path to the CSV file containing phi MRR data
        phi_eval_file_path (str): Path to the CSV file containing phi_eval MRR data
        
    Returns:
        tuple: (edge_drop_probabilities, phi_values, phi_eval_values)
            - edge_drop_probabilities (list): List of edge-drop probability values
            - phi_values (list): MRR values for phi
            - phi_eval_values (list): MRR values for phi_eval
    """
    # Read phi data
    edge_drop_prob_phi, phi_values = read_single_mrr_file(phi_file_path)
    
    # Read phi_eval data
    edge_drop_prob_phi_eval, phi_eval_values = read_single_mrr_file(phi_eval_file_path)
    
    # Verify that edge-drop probabilities match between files
    if edge_drop_prob_phi != edge_drop_prob_phi_eval:
        raise ValueError("Edge-drop probabilities do not match between phi and phi_eval files")
    
    return edge_drop_prob_phi, phi_values, phi_eval_values


def setup_plot_style():
    """
    Set up the plot style and appearance.
    
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set labels
    ax.set_xlabel('Random edge-drop probability', fontsize=12)
    ax.set_ylabel('MRR', fontsize=12)
    
    return fig, ax


def plot_mrr_curves(ax, edge_drop_probabilities, phi_values, phi_eval_values, 
                   phi_label='Phi', phi_eval_label='Phi Eval'):
    """
    Plot MRR curves for phi and phi_eval with data labels.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis object
        edge_drop_probabilities (list): List of edge-drop probability values
        phi_values (list): MRR values for phi
        phi_eval_values (list): MRR values for phi_eval
        phi_label (str, optional): Label for phi line. Defaults to 'Phi'.
        phi_eval_label (str, optional): Label for phi_eval line. Defaults to 'Phi Eval'.
        
    Returns:
        tuple: (line1, line2) matplotlib line objects for phi and phi_eval
    """
    # Plot phi line (blue circles)
    line1 = ax.plot(edge_drop_probabilities, phi_values, 
                    'o-', color='blue', linewidth=2, markersize=8, 
                    label=phi_label)
    
    # Plot phi_eval line (red squares)
    line2 = ax.plot(edge_drop_probabilities, phi_eval_values, 
                    's-', color='red', linewidth=2, markersize=8, 
                    label=phi_eval_label)
    
    # Calculate offset for label positioning to avoid overlap
    all_values = phi_values + phi_eval_values
    y_range = max(all_values) - min(all_values)
    offset = y_range * 0.05  # 5% of the data range
    
    # Add labels for phi values (positioned above the line)
    for i, (x, y) in enumerate(zip(edge_drop_probabilities, phi_values)):
        ax.text(x, y + offset, f'{y:.3f}', 
                ha='center', va='bottom', fontsize=9, 
                color='blue', rotation=45)
    
    # Add labels for phi_eval values (positioned below the line)
    for i, (x, y) in enumerate(zip(edge_drop_probabilities, phi_eval_values)):
        ax.text(x, y - offset*2, f'{y:.3f}', 
                ha='center', va='top', fontsize=9, 
                color='red', rotation=45)
    
    return line1[0], line2[0]


def finalize_plot(ax, dataset_name, phi_values, phi_eval_values, show_legend=True):
    """
    Finalize the plot with title, legend, and formatting.
    
    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis object
        dataset_name (str): Name of the dataset for the title
        phi_values (list): MRR values for phi (used for dynamic axis scaling)
        phi_eval_values (list): MRR values for phi_eval (used for dynamic axis scaling)
        show_legend (bool, optional): Whether to show legend. Defaults to True.
    """
    # Set title
    ax.set_title(f'{dataset_name}', 
                fontsize=14, fontweight='bold')
    
    # Calculate dynamic axis limits based on data
    all_values = phi_values + phi_eval_values
    min_mrr = min(all_values)
    max_mrr = max(all_values)
    
    # Add extra padding to accommodate labels above and below data points
    y_range = max_mrr - min_mrr
    label_offset = y_range * 0.05  # Same offset used for labels
    y_padding = (max_mrr - min_mrr) * 0.15  # Increased padding for labels
    y_min = max(0, min_mrr - label_offset - y_padding)
    y_max = max_mrr + label_offset + y_padding
    
    # Set axis limits
    ax.set_xlim(-0.05, 0.55)
    ax.set_ylim(y_min, y_max)
    
    # Set tick intervals
    ax.set_xticks(np.arange(0, 0.6, 0.1))
    
    # Add legend
    if show_legend:
        ax.legend(loc='upper right', fontsize=10)
    
    # Tight layout
    plt.tight_layout()


def create_mrr_plot_from_files(phi_file_path, phi_eval_file_path, dataset_name, 
                              phi_label='Phi', phi_eval_label='Phi Eval', 
                              save_path=None, show_plot=True):
    """
    Create a complete MRR plot from two separate CSV files.
    
    Args:
        phi_file_path (str): Path to the CSV file containing phi MRR data
        phi_eval_file_path (str): Path to the CSV file containing phi_eval MRR data
        dataset_name (str): Name of the dataset for the title
        phi_label (str, optional): Label for phi line. Defaults to 'Phi'.
        phi_eval_label (str, optional): Label for phi_eval line. Defaults to 'Phi Eval'.
        save_path (str, optional): Path to save the plot. If None, plot is not saved.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Read data from both files
    edge_drop_probabilities, phi_values, phi_eval_values = read_mrr_data_from_files(
        phi_file_path, phi_eval_file_path)
    
    # Setup plot
    fig, ax = setup_plot_style()
    
    # Plot curves
    plot_mrr_curves(ax, edge_drop_probabilities, phi_values, phi_eval_values,
                   phi_label=phi_label, phi_eval_label=phi_eval_label)
    
    # Finalize plot with dynamic axis scaling
    finalize_plot(ax, dataset_name, phi_values, phi_eval_values)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig, ax


if __name__ == "__main__":

    # # NELL FILES
    # phi_file_path = "./experiments/NBFNet/NELL995/2025-09-25-13-40-00-phi-finetuned/test_eval_on_edge_drop.csv"
    # phi_eval_file_path = "./experiments/NBFNet/NELL995/2025-09-25-13-41-51-phi-unif-finetuned/test_eval_on_edge_drop.csv"
    # dataset_name = "NELL995"

    # YAGO FILES
    phi_file_path = "./experiments/NBFNet/YAGO310/2025-09-23-15-31-48-phi-inf-on-rand-edge-drop-re/test_eval_on_edge_drop_re.csv"
    phi_eval_file_path = "/scratch/dd115/Graph-Transformer/NBFNet-PyG/experiments/NBFNet/YAGO310/2026-02-02-17-32-47-phi-unif-10b-e03-resumed-2026-01-29-13-1.pth/test_eval_on_edge_drop.csv"
    dataset_name = "YAGO310"
    filename = "Phi-Orig-Phi-Unif-10b-E03-Inf-On-Edge-Drop"

    # WN FILES
    # phi_file_path = "./experiments/NBFNet/WN18RR/2025-09-16-13-12-26/test_eval_on_edge_drop_re.csv"
    # phi_eval_file_path = "./experiments/NBFNet/WN18RR/2025-09-16-13-12-26/test_eval_on_edge_drop_re.csv"
    # dataset_name = "WN18RR"

    # SAVE PATH with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if filename:
        save_dir = f"./plots/raw_explainer_figures/{timestamp}-{dataset_name}-{filename}"
    else:
        save_dir = f"./plots/raw_explainer_figures/{timestamp}-{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/fig2_mrr_vs_rand_edge_drop_{dataset_name}.png"

    # Create the plot from two files
    fig, ax = create_mrr_plot_from_files(
        phi_file_path=phi_file_path,
        phi_eval_file_path=phi_eval_file_path,
        dataset_name=dataset_name,
        phi_label="Phi",
        phi_eval_label="Phi Eval",
        save_path=save_path
    )

    # Save metadata file with CSV paths
    metadata_file = f"{save_dir}/metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Plot generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Label: {filename}\n")
        f.write(f"Plot file: fig2_mrr_vs_rand_edge_drop_{dataset_name}.png\n\n")
        f.write("CSV Files Used:\n")
        f.write(f"phi_csv = \"{phi_file_path}\"\n")
        f.write(f"phi_eval_unif_csv = \"{phi_eval_file_path}\"\n")

    print(f"Plot saved to: {save_path}")
    print(f"Metadata saved to: {metadata_file}")