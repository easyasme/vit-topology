import torch
import numpy as numpy
from sklearn.decomposition import PCA

# cla to each block
def apply_cla_to_block(block_output, reduction_rate=0.5, method='random', max_iterations=1000, tolerance=1):
    # block_output: tensor of shape [batch_size, sequence_length, embedding_dimension]
    # Returns reduced_embeddings - tensor of reduced embeddings

    # Flatten data
    batch_size, sequence_length, embedding_dimension = block_output.shape
    embeddings = block_output.view(-1, embedding_dimension) # [batch_size * sequence_length, embedding_dimension]

    # PCA - dimensionality reduction
    n_components = 100 # Question - do we want to reduce it to 100?
    pca = PCA(n_components=n_components)
    embeddings_np = embeddings.cpu().numpy()
    embeddings_reduced_np = pca.fit_transform(embeddings_np)
    embeddings_reduced = torch.tensor(embeddings_reduced_np, device = embeddings.device) # numpy -> tensor

    # Find delta - iteratively till resulting the desired reduction rate
    delta = find_delta_for_reduction_rate(embeddings_reduced, reduction_rate, method, max_iterations, tolerance)

    # Grid indices calculated
    min_values = embeddings_reduced.min(dim=0)[0] # min value of data for starting point - dividing the space into grid
    # Actual split of the space -> indices - tensor that each row represents the grid indices of data point
    grid_indices_of_data_point = torch.floor((embeddings_reduced - min_values) / delta).long() # grid index for each data point in embeddings_reduced

    # Find representative point in each grid
    # group data point in each grid - unique_indices = unique grid cells where data points are located, inverse_indices = which unique grid cell each point belongs to
    # grid without data points is not included in unique_indices
    unique_indices, inverse_indices = torch.unique(grid_indices_of_data_point, dim=0, return_inverse=True)
    representatives = []

    print (unique_indices)
    print (inverse_indices)

    for idx in range(unique_indices.size(0)):
        points_in_cell = embeddings_reduced[(inverse_indices == idx)] # find points in current grid

        if method == 'random':
            representatives.append(points_in_cell[0])

        elif method == 'closest_to_mean':
            # data point closest to mean
            grid_cell_mean = points_in_cell.mean(dim=0)
            distances = torch.norm(points_in_cell - grid_cell_mean, dim=1) # Euclidean distance to mean
            closest_point = points_in_cell[distances.argmin()]
            representatives.append(closest_point)
        
        else:
            raise ValueError("Method do not exist")

    reduced_embeddings = torch.stack(representatives) # [k representative points, embedding_dimension]

    return reduced_embeddings

def find_delta_for_reduction_rate(embeddings, reduction_rate, method='random', max_iterations=1000, tolerance=1):
    # delta = side length of the hypercubes

    n_samples = embeddings.size(0)
    target_size = int(n_samples * reduction_rate)
    
    # Initialize delta guess using std
    delta = embeddings.std().item() / 10 # might need to divide by other value

    for _ in range(max_iterations):
        # Grid indices calculated
        min_values = embeddings.min(dim=0)[0]
        indices = torch.floor((embeddings - min_values) / delta).long() # data point belongs to which grid

        # Count number of unique grid cells - unique indices
        unique_indices = torch.unique(indices, dim=0)
        reduced_size = unique_indices.size(0)

        # Check reduced size -> target
        if abs(reduced_size - target_size) <= tolerance:
            break
        elif reduced_size > target_size:
            # increase delta
            delta *= 1.05
        else:
            delta /= 1.05
    
    return delta



# Test

batch_size = 20
sequence_length = 10
embedding_dimension = 768

block_output = torch.rand(batch_size, sequence_length, embedding_dimension)

reduction_rate = 0.5

reduced_embeddings_closest_to_mean = apply_cla_to_block(block_output, reduction_rate, method='closest_to_mean')

print("Original shape:", block_output.shape)
print("Reduced (closest to mean) shape:", reduced_embeddings_closest_to_mean.shape)