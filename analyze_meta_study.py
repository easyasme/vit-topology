import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def analyze_meta_study(results_dir='test', grid_samples=6):
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

        # Extract data from the results dictionary
        for key, values in results_dict.items():
            seq_len, embed_dim, delta = key
            n_samples_reduced, n_samples_og, _, _ = values

            sequence_lengths.append(seq_len)
            embedding_dimensions.append(embed_dim)
            deltas.append(delta)
            n_samples_reduced_list.append(n_samples_reduced)
            n_samples_og_list.append(n_samples_og)

        sequence_lengths = np.array(sequence_lengths)
        embedding_dimensions = np.array(embedding_dimensions)
        deltas = np.array(deltas)
        n_samples_reduced_list = np.array(n_samples_reduced_list)
        n_samples_og_list = np.array(n_samples_og_list)

        reduction_ratios = n_samples_reduced_list / n_samples_og_list

        # Unique values for plotting
        unique_embedding_dims = np.unique(embedding_dimensions)
        unique_sequence_lengths = np.unique(sequence_lengths)
        unique_deltas = np.unique(deltas)

        # Reduction Ratio vs. Embedding Dimension
        plt.figure(figsize=(10, 6))
        plt.scatter(embedding_dimensions, reduction_ratios, marker='o')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Reduction Ratio')
        plt.title(f'Reduction Ratio vs. Embedding Dimension\n{distribution.capitalize()} Distribution')
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'graph1_{distribution}.png')
        print(f"Saving graph to: {save_path}\n")
        plt.savefig(save_path)
        plt.close()

        # Reduction Ratio vs. Sequence Length
        plt.figure(figsize=(10, 6))
        plt.scatter(sequence_lengths, reduction_ratios, marker='o')
        plt.xlabel('Sequence Length')
        plt.ylabel('Reduction Ratio')
        plt.title(f'Reduction Ratio vs. Sequence Length\n{distribution.capitalize()} Distribution')
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'graph2_{distribution}.png')
        print(f"Saving graph to: {save_path}\n")
        plt.savefig(save_path)
        plt.close()

        # Reduction Ratio vs. Delta
        plt.figure(figsize=(10, 6))
        plt.scatter(deltas, reduction_ratios, marker='o')
        plt.xlabel('Delta')
        plt.ylabel('Reduction Ratio')
        plt.title(f'Reduction Ratio vs. Delta\n{distribution.capitalize()} Distribution')
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'graph3_{distribution}.png')
        print(f"Saving graph to: {save_path}\n")
        plt.savefig(save_path)
        plt.close()

        # 3D Scatter Plot with Color Mapping
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(embedding_dimensions, sequence_lengths, deltas, c=reduction_ratios, cmap='viridis', marker='o')
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Sequence Length')
        ax.set_zlabel('Delta')
        plt.title(f'Reduction Ratio for {distribution.capitalize()} Distribution')
        cbar = plt.colorbar(sc, pad=0.1)
        cbar.set_label('Reduction Ratio')
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'reduction_ratio_3d_{distribution}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"3D scatter plot saved to: {save_path}\n")

        # Heatmaps - Fixed Delta Values
        for delta_val in unique_deltas:
            indices = deltas == delta_val
            if np.any(indices):
                embed_dims = embedding_dimensions[indices]
                seq_lens = sequence_lengths[indices]
                red_ratios = reduction_ratios[indices]

                # Create grid
                embed_dim_values = np.unique(embed_dims)
                seq_len_values = np.unique(seq_lens)
                grid_embed_dim, grid_seq_len = np.meshgrid(embed_dim_values, seq_len_values)

                # Initialize grid for reduction ratios
                grid_red_ratio = np.full(grid_embed_dim.shape, np.nan)

                for i, seq_len in enumerate(seq_len_values):
                    for j, embed_dim in enumerate(embed_dim_values):
                        idx = (embed_dims == embed_dim) & (seq_lens == seq_len)
                        if np.any(idx):
                            grid_red_ratio[i, j] = red_ratios[idx][0]

                plt.figure(figsize=(10, 8))
                plt.imshow(grid_red_ratio, origin='lower', aspect='auto', extent=[embed_dim_values[0], embed_dim_values[-1], seq_len_values[0], seq_len_values[-1]], cmap='viridis')
                plt.colorbar(label='Reduction Ratio')
                plt.xlabel('Embedding Dimension')
                plt.ylabel('Sequence Length')
                plt.title(f'Reduction Ratio Heatmap\n{distribution.capitalize()} Distribution\nDelta={delta_val}')
                plt.tight_layout()
                save_path = os.path.join(results_dir, f'reduction_ratio_heatmap_{distribution}_delta{delta_val}.png')
                plt.savefig(save_path)
                plt.close()
                print(f"Heatmap saved to: {save_path}\n")

        print(f"Graphs saved in {results_dir} for distribution {distribution}.")


        # Calculate Distance Ratio (Reduced / Original)
        dist_ratio_reduced_original = reduced_dist_ratios / og_dist_ratios

        # Distance Ratio (Reduced / Original) vs Embedding Dimension
        plt.figure(figsize=(10, 6))
        for delta_val in unique_deltas:
            indices = deltas == delta_val
            if np.any(indices):
                plt.plot(embedding_dimensions[indices], dist_ratio_reduced_original[indices], marker='o', label=f'Delta: {delta_val:.2f}')
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Distance Ratio (Reduced / Original)')
        plt.title(f'Distance Ratio (Reduced / Original) vs. Embedding Dimension\n{distribution.capitalize()} Distribution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'dist_ratio_vs_embedding_{distribution}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Graph saved to: {save_path}")

        # Distance Ratio (Reduced / Original) vs Delta
        plt.figure(figsize=(10, 6))
        for embed_dim in unique_embedding_dims:
            indices = embedding_dimensions == embed_dim
            if np.any(indices):
                plt.plot(deltas[indices], dist_ratio_reduced_original[indices], marker='o', label=f'Embed Dim: {embed_dim}')
        plt.xlabel('Delta')
        plt.ylabel('Distance Ratio (Reduced / Original)')
        plt.title(f'Distance Ratio (Reduced / Original) vs. Delta\n{distribution.capitalize()} Distribution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'dist_ratio_vs_delta_{distribution}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Graph saved to: {save_path}")

        print(f"Graphs saved in {results_dir} for distribution {distribution}.")

if __name__ == "__main__":
    analyze_meta_study()
