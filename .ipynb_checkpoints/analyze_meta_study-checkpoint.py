import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def analyze_meta_study(results_dir='meta_study_results', grid_samples=6):
    distributions = ['uniform', 'normal', 'exponential', 'beta', 'log_normal', 'gamma']

    for distribution in distributions:
        print(f"\nAnalyzing distribution: {distribution}")
        filename = f"meta_study_{distribution}_gs_{grid_samples}.pkl"
        file_path = os.path.join(results_dir, filename)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'rb') as f:
            results_dict = pickle.load(f)

        sequence_lengths = []
        embedding_dimensions = []
        deltas = []
        n_samples_reduced_list = []
        n_samples_og_list = []
        reduced_dist_ratios = []
        og_dist_ratios = []

        # Extract data from the results dictionary
        for key, values in results_dict.items():
            seq_len, embed_dim, delta = key
            n_samples_reduced, n_samples_og, reduced_dist_ratio, og_dist_ratio = values

            sequence_lengths.append(seq_len)
            embedding_dimensions.append(embed_dim)
            deltas.append(delta)
            n_samples_reduced_list.append(n_samples_reduced)
            n_samples_og_list.append(n_samples_og)
            reduced_dist_ratios.append(reduced_dist_ratio)
            og_dist_ratios.append(og_dist_ratio)

        sequence_lengths = np.array(sequence_lengths)
        embedding_dimensions = np.array(embedding_dimensions)
        deltas = np.array(deltas)
        n_samples_reduced_list = np.array(n_samples_reduced_list)
        n_samples_og_list = np.array(n_samples_og_list)
        reduced_dist_ratios = np.array(reduced_dist_ratios)
        og_dist_ratios = np.array(og_dist_ratios)

        reduction_ratios = n_samples_reduced_list / n_samples_og_list

        unique_embedding_dims = np.unique(embedding_dimensions)
        unique_sequence_lengths = np.unique(sequence_lengths)
        unique_deltas = np.unique(deltas)

        for seq_len in unique_sequence_lengths:
            seq_indices = sequence_lengths == seq_len

            # Reduction Ratio vs. Delta for Different Embedding Dimensions
            plt.figure(figsize=(10, 6))
            for embed_dim in unique_embedding_dims:
                indices = seq_indices & (embedding_dimensions == embed_dim)
                if np.any(indices):
                    print(f"Reduction Ratio vs Delta - Seq Len={seq_len}, Embed Dim={embed_dim}: Deltas={deltas[indices]}, Reduction Ratios={reduction_ratios[indices]}")
                    plt.plot(deltas[indices], reduction_ratios[indices], marker='o', label=f'Embed Dim: {embed_dim}')
            plt.xlabel('Delta')
            plt.ylabel('Reduction Ratio')
            plt.title(f'Reduction Ratio vs. Delta (Seq Len={seq_len})\n{distribution.capitalize()} Distribution')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(results_dir, f'graph1_{distribution}_seq{seq_len}.png')
            print(f"Saving graph to: {save_path}\n")
            plt.savefig(save_path)
            plt.close()

            # Reduced Distance Ratio vs. Delta for Different Embedding Dimensions
            plt.figure(figsize=(10, 6))
            for embed_dim in unique_embedding_dims:
                indices = seq_indices & (embedding_dimensions == embed_dim)
                if np.any(indices):
                    print(f"Reduced Distance Ratio vs Delta - Seq Len={seq_len}, Embed Dim={embed_dim}: Deltas={deltas[indices]}, Reduced Dist Ratios={reduced_dist_ratios[indices]}")
                    plt.plot(deltas[indices], reduced_dist_ratios[indices], marker='o', label=f'Embed Dim: {embed_dim}')
            plt.xlabel('Delta')
            plt.ylabel('Reduced Distance Ratio')
            plt.title(f'Reduced Distance Ratio vs. Delta (Seq Len={seq_len})\n{distribution.capitalize()} Distribution')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(results_dir, f'graph2_{distribution}_seq{seq_len}.png')
            print(f"Saving graph to: {save_path}\n")
            plt.savefig(save_path)
            plt.close()

            # Reduction Ratio vs. Embedding Dimension for Different Deltas
            plt.figure(figsize=(10, 6))
            for delta_val in unique_deltas:
                indices = seq_indices & (deltas == delta_val)
                if np.any(indices):
                    print(f"Reduction Ratio vs Embedding Dimension - Seq Len={seq_len}, Delta={delta_val}: Embedding Dims={embedding_dimensions[indices]}, Reduction Ratios={reduction_ratios[indices]}")
                    plt.plot(embedding_dimensions[indices], reduction_ratios[indices], marker='o', label=f'Delta: {delta_val:.2f}')
            plt.xlabel('Embedding Dimension')
            plt.ylabel('Reduction Ratio')
            plt.title(f'Reduction Ratio vs. Embedding Dimension (Seq Len={seq_len})\n{distribution.capitalize()} Distribution')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(results_dir, f'graph3_{distribution}_seq{seq_len}.png')
            print(f"Saving graph to: {save_path}\n")
            plt.savefig(save_path)
            plt.close()

        print(f"Graphs saved in {results_dir} for distribution {distribution}.")

if __name__ == "__main__":
    analyze_meta_study()
