
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def load_and_extract_mrr(csv_file):
    """
    Load CSV file and extract MRR values for each distance.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        tuple: (distances, mrr_values)
    """

    df = pd.read_csv(csv_file, index_col=0)
    mrr_row = df.loc['mrr']
    distances = [int(col) for col in df.columns]
    mrr_values = mrr_row.values
    return distances, mrr_values

def plot_mrr_vs_distance(phi_file, phi_eval_unif_file, phi_eval_dist_file, dataset_name, label=None, output_file=None, log_scale=True):
    """
    Create MRR vs Distance plot for three different methods with data labels.
    
    Args:
        phi_file (str): Path to φ CSV file
        phi_eval_unif_file (str): Path to φ_eval_unif CSV file  
        phi_eval_dist_file (str): Path to φ_eval_dist CSV file
        dataset_name (str): Name of the dataset for the plot title.
        label (str): Label for the plot.
        output_file (str): Optional path to save the plot.
        log_scale (bool): If True, use a logarithmic y-axis. Defaults to True.
    """
    
    # Load data from all three files
    distances_phi, mrr_phi = load_and_extract_mrr(phi_file)
    distances_unif, mrr_unif = load_and_extract_mrr(phi_eval_unif_file)
    distances_dist, mrr_dist = load_and_extract_mrr(phi_eval_dist_file)

    # Create the plot
    title=f'{dataset_name} ({label})' if label else f'{dataset_name}'
    plt.figure(figsize=(12, 7)) # Made figure slightly wider
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Plot lines with markers
    plt.plot(distances_phi, mrr_phi, 'bo-', label='φ', linewidth=2, markersize=8, zorder=5)
    plt.plot(distances_unif, mrr_unif, 'rs-', label='$\\varphi^{unif}_{eval}$', linewidth=2, markersize=8)
    plt.plot(distances_dist, mrr_dist, 'c^-', label='$\\varphi^{dist}_{eval}$', linewidth=2, markersize=8) # Changed to cyan triangle-up
    
    # LOG SCALE: Use multiplicative offsets to avoid label overlap
    if log_scale:
        
        # Add labels for the plot lines
        for i, txt in enumerate(mrr_phi):
            plt.text(distances_phi[i], mrr_phi[i] * 1.2, f'{txt:.6f}', ha='center', va='bottom', fontsize=9, color='blue', rotation=45)
        for i, txt in enumerate(mrr_unif):
            plt.text(distances_unif[i], mrr_unif[i] * 0.5, f'{txt:.6f}', ha='center', va='top', fontsize=9, color='red', rotation=45)
        for i, txt in enumerate(mrr_dist):
            plt.text(distances_dist[i], mrr_dist[i] + 0.1, f'{txt:.3f}', ha='center', va='bottom', fontsize=9, color='darkcyan')
        
        plt.yscale('log')
        plt.ylabel('MRR (log scale)', fontsize=14)
        plt.grid(True, which="both", linestyle=':', linewidth=0.7)
        
    # LINEAR SCALE: Use additive offsets to separate overlapping labels
    else:

        # Offset for the labels
        all_mrr = np.concatenate([mrr_phi, mrr_unif, mrr_dist])
        offset = (all_mrr.max() - all_mrr.min()) * 0.04

        # Add labels for the plot lines
        for i, txt in enumerate(mrr_phi):
            plt.text(distances_phi[i], mrr_phi[i] + offset*2.5, f'{txt:.3f}', ha='center', va='bottom', fontsize=9, color='blue', rotation=45)
        for i, txt in enumerate(mrr_unif):
            plt.text(distances_unif[i], mrr_unif[i] - offset*3, f'{txt:.3f}', ha='center', va='top', fontsize=9, color='red', rotation=45)
        for i, txt in enumerate(mrr_dist):
            plt.text(distances_dist[i], mrr_dist[i] + offset, f'{txt:.3f}', ha='center', va='bottom', fontsize=9, color='darkcyan')

        plt.ylabel('MRR', fontsize=14)
        plt.grid(True, linestyle=':', linewidth=0.7, alpha=0.6)

    # Common styling
    plt.xlabel('Distance', fontsize=14)
    
    # Corrected LaTeX for legend labels
    plt.legend(handles=[plt.Line2D([0], [0], color='blue', marker='o', linestyle='-', label='$\\varphi$'),
                        plt.Line2D([0], [0], color='red', marker='s', linestyle='-', label='$\\varphi^{unif}_{eval}$'),
                        plt.Line2D([0], [0], color='cyan', marker='^', linestyle='-', label='$\\varphi^{dist}_{eval}$')],
               fontsize=14)
    
    all_distances = sorted(set(distances_phi + distances_unif + distances_dist))
    plt.xticks(all_distances, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout(pad=1.5)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    
    # CSV Files
    # phi_csv = "experiments/NBFNet/NELL995/2025-09-26-10-34-45-phi-finetuned-inf-on-dist-drop/test_eval_on_distance_drop.csv"
    # phi_eval_unif_csv = "experiments/NBFNet/NELL995/2025-09-26-10-31-21-phi-unif-finetuned-inf-on-dist-drop/test_eval_on_distance_drop.csv"
    # phi_eval_dist_csv = "experiments/NBFNet/NELL995/2025-09-25-16-24-03-phi-dist-finetuned/test_eval_on_distance_drop.csv"
    # dataset_name = "NELL995"

    phi_csv = "experiments/NBFNet/YAGO310/2025-09-24-01-53-20-phi-inf-on-dist-drop/test_eval_on_distance_drop.csv"
    phi_eval_unif_csv = "/scratch/dd115/Graph-Transformer/NBFNet-PyG/experiments/NBFNet/YAGO310/2026-02-04-01-48-43-phi-unif-10b-e03-inf-on-dist-drop/test_eval_on_distance_drop.csv"
    phi_eval_dist_csv = "/scratch/sl9030/gml/Graph-Transformer/NBFNet-PyG/experiments/NBFNet/YAGO310/2026-02-01-18-39-37-phi-dist-e02-inf-on-dist-drop/test_eval_on_distance_drop.csv"
    dataset_name = "YAGO310"
    filename = "Phi-Orig-Phi-Unif-10b-E03-Phi-Dist-E02"

    # SAVE PATH with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if filename:
        save_dir = f"./plots/raw_explainer_figures/{timestamp}-{dataset_name}-{filename}"
    else:
        save_dir = f"./plots/raw_explainer_figures/{timestamp}-{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the plot with a linear scale
    label = None
    # label = filename
    save_path_linear = f"{save_dir}/fig3_mrr_vs_distance_{dataset_name}_linear.png"
    plot_mrr_vs_distance(phi_csv, phi_eval_unif_csv, phi_eval_dist_csv, dataset_name, label, save_path_linear, log_scale=False)
    
    # Save metadata file with CSV paths
    metadata_file = f"{save_dir}/metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"Plot generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Label: {filename}\n")
        f.write(f"Scale: linear\n")
        f.write(f"Plot file: fig3_mrr_vs_distance_{dataset_name}_linear.png\n\n")
        f.write("CSV Files Used:\n")
        f.write(f"phi_csv = \"{phi_csv}\"\n")
        f.write(f"phi_eval_unif_csv = \"{phi_eval_unif_csv}\"\n")
        f.write(f"phi_eval_dist_csv = \"{phi_eval_dist_csv}\"\n")
    
    print(f"Plot saved to: {save_path_linear}")
    print(f"Metadata saved to: {metadata_file}")
    print("\nAll complete!")
