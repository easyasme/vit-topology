import torch
import numpy as np
import pickle
import os
import argparse

@torch.no_grad()
def perform_cla(features, reduction_rate=0.5, method='random', max_iterations=1000, tolerance=1, device_list=None, pca=False, pre_delta=None):
    # features: tensor of shape [batch_size, sequence_length, embedding_dimension]
    # Returns reduced_embeddings - tensor of reduced embeddings

    def find_delta(embeddings, reduction_rate, method='random', max_iterations=1000, tolerance=1):
        n_samples = embeddings.size(0)
        target_size = int(n_samples * reduction_rate)
        
        # Initialize delta guess using std
        std = embeddings.std(dim=0).min()
        print(f'std size: {std.size()}')
        delta = std.item() / 10 # might need to divide by other value

        for i in range(max_iterations):
            # Grid indices calculated
            min_values = embeddings.min(dim=0)[0]
            indices = torch.floor((embeddings - min_values) / delta).long() # data point belongs to which grid

            # Count number of unique grid cells - unique indices
            unique_indices = torch.unique(indices, dim=0)
            reduced_size = unique_indices.size(0)

            # Check reduced size -> target
            if abs(reduced_size - target_size) <= tolerance:
                print(f'iter: {i}')
                break
            elif reduced_size > target_size:
                # increase delta
                delta *= 1.1
            else:
                delta /= 1.1
            # print(f'iter: {i}')
        
        return delta

    # Flatten data
    batch_size, sequence_length, embedding_dimension = features.size()
    embeddings = features.view(-1, embedding_dimension).to(device_list[0]) # [batch_size * patches, embedding_dimension]

    # PCA - dimensionality reduction among embedding dimension
    if pca:
        embeddings_reduced_np = perform_pca(embeddings, alpha=.05, center_only=True, device_list=device_list)
        embeddings_reduced = torch.tensor(embeddings_reduced_np, device=device_list[0]) # numpy -> tensor

    # Find delta - iteratively till resulting the desired reduction rate
    if pre_delta is None:
        delta = find_delta(embeddings, reduction_rate, method, max_iterations, tolerance) # side length of hypercubes
    else:
        delta = pre_delta
    print(f'delta: {delta}')

    # Grid indices calculated
    min_values = embeddings.min(dim=0)[0] # min value of data for starting point - dividing the space into grid
    # print(f'minned vals: {embeddings - min_values}')

    # Actual split of the space -> indices - tensor that each row represents the grid indices of data point
    grid_indices_of_data_point = torch.floor((embeddings - min_values) / delta).to(device_list[-1]).long() # grid index for each data point in embeddings_reduced
    # print(f'indices: {grid_indices_of_data_point}')

    # Find representative point in each grid
    # group data point in each grid - unique_indices = unique grid cells where data points are located, inverse_indices = which unique grid cell each point belongs to
    # grid without data points is not included in unique_indices
    unique_indices, inverse_indices = torch.unique(grid_indices_of_data_point, dim=0, return_inverse=True)
    representatives = []

    print (f'unique: {unique_indices.size()}')
    print (f'inverse: {inverse_indices.size()}\n')

    for idx in range(unique_indices.size(0)):
        points_in_cell = embeddings[(inverse_indices == idx)] # find points in current grid
        # print(f'points in cell: {points_in_cell}')
        if method == 'random':
            representatives.append(points_in_cell[0])
        elif method == 'closest_to_mean':
            # data point closest to mean
            grid_cell_mean = points_in_cell.mean(dim=0)
            distances = torch.norm(points_in_cell - grid_cell_mean, dim=1) # Euclidean distance to mean
            closest_point = points_in_cell[distances.argmin()]
            representatives.append(closest_point)
        else:
            raise ValueError("Method does not implemented")

    reduced_embeddings = torch.stack(representatives) # [k representative points, embedding_dimension]

    return reduced_embeddings


# Generate data with specified distribution
def generate_data(batch_size, sequence_length, embedding_dimension, distribution='uniform', device=None):
    if distribution == 'uniform':
        data = torch.rand(batch_size, sequence_length, embedding_dimension, device=device)
        
    elif distribution == 'normal':
        data = torch.randn(batch_size, sequence_length, embedding_dimension, device=device)
        
    elif distribution == 'binary':
        data = torch.randint(0, 2, (batch_size, sequence_length, embedding_dimension), device=device, dtype=torch.float32)
        
    elif distribution == 'exponential':
        data = torch.distributions.Exponential(1.0).sample((batch_size, sequence_length, embedding_dimension)).to(device)
        
    elif distribution == 'beta':
        data = torch.distributions.Beta(2.0, 5.0).sample((batch_size, sequence_length, embedding_dimension)).to(device)
        
    elif distribution == 'log_normal':
        data = torch.distributions.LogNormal(0.0, 1.0).sample((batch_size, sequence_length, embedding_dimension)).to(device)
        
    elif distribution == 'poisson':
        # Lambda = 5
        data = torch.poisson(torch.full((batch_size, sequence_length, embedding_dimension), 5.0, device=device))
        
    elif distribution == 'gamma':
        # Alpha beta = 2.0
        data = torch.distributions.Gamma(2.0, 2.0).sample((batch_size, sequence_length, embedding_dimension)).to(device)
        
    else:
        raise ValueError("Unsupported distribution type")
    
    return data

def run_meta_study():
    parser = argparse.ArgumentParser(description='Run CLA Meta-Study')
    parser.add_argument('--study', type=str, required=True, choices=['embedding_dimension', 'sequence_length', 'reduction_rate'], help='Parameter to study')
    parser.add_argument('--output_dir', type=str, default='meta_study_results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    batch_size = 12
    tolerance = 1
    max_iterations = 1000
    method = 'random'
    distributions = ['uniform', 'normal', 'binary', 'exponential', 'beta', 'log_normal', 'poisson', 'gamma']

    # small, medium, big values
    embedding_dimension_values = [50, 100, 200, 400, 800, 1000]
    sequence_length_values = [50, 100, 200, 400, 800, 1000]
    reduction_rate_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

    if args.study == 'embedding_dimension':
        varying_param = list(range(1, 1001))
        varying_param_name = 'embedding_dimension'
        for embedding_dimension in varying_param:
            for sequence_length in sequence_length_values:
                for reduction_rate in reduction_rate_values:
                    for distribution in distributions:
                        data = generate_data(batch_size, sequence_length, embedding_dimension, distribution=distribution, device=args.device)

                        reduced_embeddings, delta = perform_cla(
                            data,
                            reduction_rate=reduction_rate,
                            method=method,
                            max_iterations=max_iterations,
                            tolerance=tolerance,
                            device=args.device,
                            pca=False
                        )

                        # Calculate actual reduction rate
                        n_samples_original = data.view(-1, embedding_dimension).size(0)
                        n_samples_reduced = reduced_embeddings.size(0)
                        actual_reduction_rate = n_samples_reduced / n_samples_original

                        # Save results
                        params = {
                            'embedding_dimension': embedding_dimension,
                            'sequence_length': sequence_length,
                            'reduction_rate': reduction_rate,
                            'method': method,
                            'distribution': distribution,
                            'delta': delta,
                            'actual_reduction_rate': actual_reduction_rate
                        }
                        # study_embedding_dimension_ed100_sl50_rr50_duniform
                        filename = f"study_{args.study}_ed{embedding_dimension}_sl{sequence_length}_rr{int(reduction_rate*100)}_d{distribution}.pkl"
                        save_path = os.path.join(args.output_dir, filename)

                        with open(save_path, 'wb') as f:
                            pickle.dump(params, f)

                        print(f"Saved results to {save_path}")

    elif args.study == 'sequence_length':
        varying_param = list(range(1, 1001))
        varying_param_name = 'sequence_length'
        for sequence_length in varying_param:
            for embedding_dimension in embedding_dimension_values:
                for reduction_rate in reduction_rate_values:
                    for distribution in distributions:
                        data = generate_data(batch_size, sequence_length, embedding_dimension, distribution=distribution, device=args.device)

                        reduced_embeddings, delta = perform_cla(
                            data,
                            reduction_rate=reduction_rate,
                            method=method,
                            max_iterations=max_iterations,
                            tolerance=tolerance,
                            device=args.device,
                            pca=False
                        )

                        n_samples_original = data.view(-1, embedding_dimension).size(0)
                        n_samples_reduced = reduced_embeddings.size(0)
                        actual_reduction_rate = n_samples_reduced / n_samples_original

                        params = {
                            'embedding_dimension': embedding_dimension,
                            'sequence_length': sequence_length,
                            'reduction_rate': reduction_rate,
                            'method': method,
                            'distribution': distribution,
                            'delta': delta,
                            'actual_reduction_rate': actual_reduction_rate
                        }
                        filename = f"study_{args.study}_ed{embedding_dimension}_sl{sequence_length}_rr{int(reduction_rate*100)}_d{distribution}.pkl"
                        save_path = os.path.join(args.output_dir, filename)

                        with open(save_path, 'wb') as f:
                            pickle.dump(params, f)

                        print(f"Saved results to {save_path}")

    elif args.study == 'reduction_rate':
        varying_param = np.arange(0, 1.05, 0.05).tolist()
        varying_param_name = 'reduction_rate'
        for reduction_rate in varying_param:
            for embedding_dimension in embedding_dimension_values:
                for sequence_length in sequence_length_values:
                    for distribution in distributions:
                        data = generate_data(batch_size, sequence_length, embedding_dimension, distribution=distribution, device=args.device)

                        reduced_embeddings, delta = perform_cla(
                            data,
                            reduction_rate=reduction_rate,
                            method=method,
                            max_iterations=max_iterations,
                            tolerance=tolerance,
                            device=args.device,
                            pca=False
                        )

                        n_samples_original = data.view(-1, embedding_dimension).size(0)
                        n_samples_reduced = reduced_embeddings.size(0)
                        actual_reduction_rate = n_samples_reduced / n_samples_original

                        params = {
                            'embedding_dimension': embedding_dimension,
                            'sequence_length': sequence_length,
                            'reduction_rate': reduction_rate,
                            'method': method,
                            'distribution': distribution,
                            'delta': delta,
                            'actual_reduction_rate': actual_reduction_rate
                        }
                        filename = f"study_{args.study}_ed{embedding_dimension}_sl{sequence_length}_rr{int(reduction_rate*100)}_d{distribution}.pkl"
                        save_path = os.path.join(args.output_dir, filename)

                        with open(save_path, 'wb') as f:
                            pickle.dump(params, f)

                        print(f"Saved results to {save_path}")
    else:
        raise ValueError("Invalid study type.")

if __name__ == "__main__":
    run_meta_study()





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
