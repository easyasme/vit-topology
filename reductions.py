import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def perform_pca(features, alpha=.05, center_only=True, device_list=None):
    ''' Perform a torch implemented GPU accelerated PCA on the features
        and return the reduced unnormalized features. Expected input shape 
        is (samples, features).
    '''
    features = torch.tensor(features, requires_grad=False).detach().to(device_list[-1]).T # features x samples
    # features = features[torch.where(features.std(dim=-1, keepdim=True)!=0)[0], :] # filter out constant rows
    # features = (features - features.mean(dim=-1, keepdim=True)) / features.std(dim=-1, keepdim=True) # standardize

    # Perform PCA
    _, S, V = torch.linalg.svd(torch.cov(features), driver='gesvd')
    S, V = S.detach().numpy(force=True), V.detach().to(device_list[-1])

    # Calculate the number of principal components to keep
    explained = S / sum(S) # calculate the percentage of variance explained by each component
    
    num_components = 0
    partial_perc = 0
    for perc in explained:
        partial_perc += perc
        num_components += 1
        if partial_perc >= 1 - alpha:
            break
    
    print(f'Explained variance: {partial_perc:.3f} with {num_components} components\n')

    # Project the data onto the principal components
    features = (features.T @ V[:, :num_components]).detach()

    # free up memory on the GPU
    del explained, S, V, num_components, partial_perc
    torch.cuda.empty_cache()

    return features.cpu().data.numpy().astype(np.float64)
    
@torch.no_grad()
def perform_kmeans(features, num_max_clusters=1000, device_list=None, metric='correlation', exp=1, corr='pearson'):
    ''' Perform a torch implemented GPU accelerated kmeans on the features and return the cluster assignments. Expected input shape is (samples, features).
    '''
    from kmeans_pytorch import find_best_cluster

    features = torch.tensor(features, requires_grad=False).detach().to(device_list[-1]).T # features x samples
    features = features[torch.where(features.std(dim=-1, keepdim=True)!=0)[0], :] # filter out constant rows

    num_max_clusters = min(num_max_clusters, features.shape[0])
    features = find_best_cluster(features, num_min_clusters=num_max_clusters, num_max_clusters=num_max_clusters, distance=metric, device=device_list[-1], tqdm_flag=True, sil_score=False, seed=SEED, corr=corr, exp=exp, minibatch=None)

    features = features[torch.where(features.std(dim=-1, keepdim=True)!=0)[0], :] # filter out constant rows
    features = (features - features.mean(dim=-1, keepdim=True)) / features.std(dim=-1, keepdim=True) # standardize

    del num_max_clusters

    return features.T.cpu().data.numpy().astype(np.float64)

@torch.no_grad()
def perform_umap(features, num_components=2, num_neighbors=50, min_dist=0.1, num_epochs=10, metric='euclidean', device_list=None):
    ''' Perform UMAP on the features.
        Possible metrics: 'euclidean', 'manhattan', 'cosine', 'hamming', 'jaccard', 'dice', 'correlation',
        'mahalanobis', 'braycurtis', 'canberra', 'chebyshev', 'rogerstanimoto'.
    '''
    import torch.nn.functional as F
    from umap_pytorch import PUMAP

    features = torch.tensor(features, requires_grad=False)

    pumap = PUMAP(
        encoder=None, # nn.Module, None for default
        decoder=None, # nn.Module, True for default, None for encoder only
        n_neighbors=num_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=num_components,
        beta=1.0, # How much to weigh reconstruction loss for decoder
        reconstruction_loss=F.binary_cross_entropy_with_logits, # pass in custom reconstruction loss functions
        random_state=SEED,
        lr=1e-3,
        epochs=num_epochs,
        batch_size=64,
        num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 1,
        num_gpus=len(device_list) if device_list is not None else 0,
        match_nonparametric_umap=False # Train network to match embeddings from non parametric umap
    )

    pumap.fit(features)
    features = pumap.transform(features)

    return features

@torch.no_grad()
def perform_cla(features, reduction_rate=0.5, method='random', max_iterations=1000, tolerance=1, device_list=None, pca=False, pre_delta=None):
    # features: tensor of shape [batch_size, sequence_length, embedding_dimension]
    # Returns reduced_embeddings - tensor of reduced embeddings

    def find_delta(embeddings, reduction_rate, max_iterations=1000, tolerance=1):
        n_samples = embeddings.size(0)
        target_size = int(n_samples * reduction_rate)
        
        # Initialize delta guess using std
        std = embeddings.std(dim=0).min()
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
                break
            elif reduced_size > target_size:
                # increase delta
                delta *= 1.1
            else:
                delta /= 1.1
        
        return delta

    # Flatten data
    _, _, embedding_dimension = features.size()
    embeddings = features.view(-1, embedding_dimension).to(device_list[0]) # [batch_size * patches, embedding_dimension]

    # PCA - dimensionality reduction among embedding dimension
    if pca:
        embeddings_reduced_np = perform_pca(embeddings, alpha=.05, center_only=True, device_list=device_list)
        embeddings = torch.tensor(embeddings_reduced_np, device=device_list[0]) # numpy -> tensor

    # Find delta - iteratively till resulting the desired reduction rate
    if pre_delta is None:
        delta = find_delta(embeddings, reduction_rate, method, max_iterations, tolerance) # side length of hypercubes
    else:
        delta = pre_delta

    # Grid indices calculated
    min_values = embeddings.min(dim=0)[0] # min value of data for starting point - dividing the space into grid

    # Actual split of the space -> indices - tensor that each row represents the grid indices of data point
    grid_indices_of_data_point = torch.floor((embeddings - min_values) / delta).to(device_list[-1]).long()

    # Find representative point in each grid
    # group data point in each grid - unique_indices = unique grid cells where data points are located, inverse_indices 
    # is which unique grid cell each point belongs to
    # grid without data points is not included in unique_indices
    unique_indices, inverse_indices = torch.unique(grid_indices_of_data_point, dim=0, return_inverse=True)
    unique_indices, inverse_indices = unique_indices.to(device_list[-1]), inverse_indices.to(device_list[0])
    representatives = []

    for idx in range(unique_indices.size(0)):
        points_in_cell = embeddings[(inverse_indices == idx)] # find points in current grid
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

    return reduced_embeddings, delta
