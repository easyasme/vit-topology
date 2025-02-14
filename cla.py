import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import argparse

from reductions import perform_cla


# Generate data with specified distribution
def generate_data(batch_size, sequence_length, embedding_dimension, distribution='uniform', device_list=None):
    if distribution == 'uniform':
        data = torch.rand(size=(batch_size, sequence_length, embedding_dimension), device=device_list[-1])
        
    elif distribution == 'normal':
        data = torch.randn(batch_size, sequence_length, embedding_dimension, device=device_list[-1])
        
    elif distribution == 'binary':
        data = torch.randint(0, 2, (batch_size, sequence_length, embedding_dimension), device=device_list[-1], dtype=torch.float32)
        
    elif distribution == 'exponential':
        data = torch.distributions.Exponential(1.0).sample((batch_size, sequence_length, embedding_dimension)).to(device_list[-1])
        
    elif distribution == 'beta':
        data = torch.distributions.Beta(2.0, 5.0).sample((batch_size, sequence_length, embedding_dimension)).to(device_list[-1])
        
    elif distribution == 'log_normal':
        data = torch.distributions.LogNormal(0.0, 1.0).sample((batch_size, sequence_length, embedding_dimension)).to(device_list[-1])
        
    elif distribution == 'poisson':
        # Lambda = 5
        data = torch.poisson(torch.full((batch_size, sequence_length, embedding_dimension), 5.0, device=device_list[-1]))
        
    elif distribution == 'gamma':
        # Alpha beta = 2.0
        data = torch.distributions.Gamma(2.0, 2.0).sample((batch_size, sequence_length, embedding_dimension)).to(device_list[-1])
        
    else:
        raise ValueError("Unsupported distribution type")
    
    return data

def calculate_distance_ratio(x): # Pairwise distances; returns upper triangular matrix; x: [N, D]
    dist = F.pdist(x)
    ratio = dist.max() / dist.min()

    del dist

    return ratio

def run_meta_study(args):
    os.makedirs(args.output_dir, exist_ok=True)

    batch_size = 10
    tolerance = 1
    max_iterations = 1000
    method = 'closest_to_mean' # 'random'
    pca = False

    # small, medium, big values
    distributions = ['uniform', 'normal', 'exponential', 'beta', 'log_normal', 'gamma']
    embedding_dimension_values = np.linspace(2, 20, args.grid_samples, dtype=int) # [50, 100, 200, 400, 800, 1000]
    sequence_length_values = np.linspace(100, 2000, args.grid_samples, dtype=int) # [50, 100, 200, 400, 800, 1000]
    pre_delta_values = np.linspace(.01, 1, args.grid_samples) # [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

    for distribution in distributions:
        results_dict = dict()
        print(f"\nRunning meta-study for distribution: {distribution}")
        for sequence_length in sequence_length_values:
            for embedding_dimension in embedding_dimension_values:
                data = generate_data(batch_size,
                                     sequence_length,
                                     embedding_dimension,
                                     distribution=distribution,
                                     device_list=args.device_list
                                    )
                for delta in pre_delta_values:
                    reduced_embeddings, delta = perform_cla(data,
                                                            method=method,
                                                            max_iterations=max_iterations,
                                                            tolerance=tolerance,
                                                            device_list=args.device_list,
                                                            pca=pca,
                                                            pre_delta=delta
                                                           )

                    # Calculate actual reduction rate
                    n_samples_reduced = reduced_embeddings.size(0)
                    n_samples_og = data.view(-1, embedding_dimension).size(0)

                    if n_samples_reduced >= 2:
                        reduced_dist_ratio = calculate_distance_ratio(reduced_embeddings).numpy(force=True)
                        og_dist_ratio = calculate_distance_ratio(data.view(-1, embedding_dimension)).numpy(force=True)
                    else:
                        reduced_dist_ratio = np.nan
                        og_dist_ratio = np.nan

                    # Save results
                    results = np.array([n_samples_reduced, n_samples_og, reduced_dist_ratio, og_dist_ratio])
                    results_dict[(sequence_length, embedding_dimension, delta)] = results

        filename = f"meta_study_{distribution}_gs_{args.grid_samples}.pkl"
        save_path = os.path.join(args.output_dir, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL) # python 3.8.18

        print(f"Saved results to {save_path}")

if __name__ == "__main__":
    device_list = []
    if torch.cuda.device_count() > 1:
        device_list = [torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count())]
        print("\nUsing", torch.cuda.device_count(), "GPUs")
        for i, device in enumerate(device_list):
            print(f"Device {i}: {device}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_list.append(device)
        print(f'Using {device}')

    parser = argparse.ArgumentParser(description='Run CLA Meta-Study')

    parser.add_argument('--output_dir', type=str, default='meta_study_results')
    parser.add_argument('--grid_samples', type=int, default=6)
    parser.add_argument('--device_list', type=list, default=device_list)
    
    args = parser.parse_args()
    run_meta_study(args)


# import torch
# import numpy as np

# def apply_cla_to_block(block_output, reduction_rate=0.5, method='random', max_iterations=1000, tolerance=1):
#     # Flatten data
#     batch_size, sequence_length, embedding_dimension = block_output.shape
#     embeddings = block_output.view(-1, embedding_dimension)  # [N, D]

#     # Normalize the data to [0, 1]
#     min_values = embeddings.min(dim=0)[0]
#     max_values = embeddings.max(dim=0)[0]
#     embeddings_normalized = (embeddings - min_values) / (max_values - min_values + 1e-8)

#     # Find delta - adjusted strategy
#     delta = find_delta_for_reduction_rate(
#         embeddings_normalized, reduction_rate, method, max_iterations, tolerance
#     )
#     print(f'Delta found: {delta}')

#     # Compute grid indices
#     grid_indices_of_data_point = torch.floor(embeddings_normalized / delta).long()

#     # Find representative point in each grid
#     unique_indices, inverse_indices = torch.unique(grid_indices_of_data_point, dim=0, return_inverse=True)
#     representatives = []

#     for idx in range(unique_indices.size(0)):
#         points_in_cell = embeddings[(inverse_indices == idx)]  # Use original embeddings

#         if method == 'random':
#             representatives.append(points_in_cell[0])
#         elif method == 'closest_to_mean':
#             grid_cell_mean = points_in_cell.mean(dim=0)
#             distances = torch.norm(points_in_cell - grid_cell_mean, dim=1)
#             closest_point = points_in_cell[distances.argmin()]
#             representatives.append(closest_point)
#         else:
#             raise ValueError("Method does not exist")

#     reduced_embeddings = torch.stack(representatives)

#     # Calculate and print the actual reduction rate
#     n_samples_original = embeddings.size(0)
#     n_samples_reduced = reduced_embeddings.size(0)
#     actual_reduction_rate = n_samples_reduced / n_samples_original
#     print(f"Actual reduction rate: {actual_reduction_rate:.4f}")

#     return reduced_embeddings

# def find_delta_for_reduction_rate(embeddings, reduction_rate, method='random', max_iterations=1000, tolerance=1):
#     n_samples = embeddings.size(0)
#     target_size = int(n_samples * reduction_rate)
#     if target_size <= 0:
#         raise ValueError("Reduction rate too small, resulting in zero samples.")

#     # Initialize delta bounds
#     delta_min = 1e-6
#     delta_max = 1.0
#     delta = (delta_min + delta_max) / 2
#     print(f'Initial delta: {delta}')

#     for i in range(max_iterations):
#         # Compute grid indices
#         indices = torch.floor(embeddings / delta).long()
#         unique_indices = torch.unique(indices, dim=0)
#         reduced_size = unique_indices.size(0)

#         # Check reduced size against target
#         if abs(reduced_size - target_size) <= tolerance:
#             print(f'Converged at iteration {i}')
#             break
#         elif reduced_size > target_size:
#             # Increase delta to reduce more points
#             delta_min = delta
#         else:
#             # Decrease delta to reduce fewer points
#             delta_max = delta
#         delta = (delta_min + delta_max) / 2
#         if i % 10 == 0:
#             print(f'Iteration {i}: delta={delta}, reduced_size={reduced_size}, target_size={target_size}')
#     else:
#         print("Warning: Maximum iterations reached. Delta may not achieve the exact reduction rate.")

#     return delta

# # Test
# batch_size = 12
# sequence_length = 200
# embedding_dimension = 800

# block_output = torch.rand(batch_size, sequence_length, embedding_dimension)
# reduction_rate = 0.5

# reduced_embeddings_closest_to_mean = apply_cla_to_block(block_output, reduction_rate, method='closest_to_mean')

# print("Original shape:", block_output.shape)
# print("Reduced (closest to mean) shape:", reduced_embeddings_closest_to_mean.shape)