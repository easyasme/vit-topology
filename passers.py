import os

import numpy as np
import torch

from config import SEED
from graph import signal_concat
from utils import progress_bar


def get_accuracy(predictions, targets):
    ''' Compute accuracy of predictions to targets. max(predictions) is best'''
    _, predicted = predictions.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()

    return 100. * (correct / total)


class Passer():
    def __init__(self, net, loader, criterion, device, repeat=1):
        self.network = net
        self.criterion = criterion
        self.device = device
        self.loader = loader
        self.repeat = repeat

    def _pass(self, optimizer=None, mask=None):
        ''' Main data passing routing '''
        losses, features, total, correct = [], [], 0, 0
        accuracies = []
        
        for r in range(1, self.repeat + 1):
            for batch_idx, (inputs, targets) in enumerate(self.loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            
                if optimizer: 
                    optimizer.zero_grad()

                if mask:
                    outputs = self.network(inputs, mask)
                else:
                    outputs = self.network(inputs)

                loss = self.criterion(outputs, targets)
                losses.append(loss.item())

                if optimizer is not None:
                    loss.backward()
                    optimizer.step()

                
                accuracies.append(get_accuracy(outputs, targets))
                progress_bar((r-1)*len(self.loader)+batch_idx, r*len(self.loader), 'repeat %d -- Mean Loss: %.3f | Last Loss: %.3f | Acc: %.3f%%' % (r, np.mean(losses), losses[-1], np.mean(accuracies)))

        return np.asarray(losses), np.mean(accuracies)

    def get_sample(self):
        iterator = iter(self.loader)
        inputs, _ = iterator.next()

        return inputs[0:1,...].to(self.device)

    def run(self, optimizer=None, mask=None):
        if optimizer:
            self.network.train()

            return self._pass(optimizer, mask=mask)
        else:
            self.network.eval()

            with torch.no_grad():
                return self._pass(mask=mask)

    def get_predictions(self):
        ''' Returns predictions and targets '''
        preds, gts = [], []

        for batch_idx, (inputs, targets) in enumerate(self.loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.network(inputs)
                
            gts.append(targets.cpu().data.numpy())
            preds.append(outputs.cpu().data.numpy().argmax(1))
            
        return np.concatenate(gts), np.concatenate(preds)

    @torch.no_grad()
    def get_function(self, reduction=None, device_list=None):
        ''' Collect function (features) from the self.network.module.forward_features() routine '''
        features = []

        for batch_idx, (inputs, targets) in enumerate(self.loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Append to features all of the outputs of the forward_features function
            # for each data point in the batch. Note that the forward_features function
            # returns a list of tensors, so we need to iterate through the list and
            # append each tensor to the features list as a numpy array of type float16.
            # Further note that the tensors within the list of tensors, in the case of the 
            # 3 layer FCNet, are of size 100x3 and 100x4, respectively, where 100 is the
            # batch size and the second dimension is the number of neurons in the layer.

            features.append([f.cpu().data.numpy().astype(np.float32) for f in self.network.forward_features(inputs)])
                
            progress_bar(batch_idx, len(self.loader))

        # So for each data point in the batch, we have a 3x1 and 4x1 vector of activations
        # for the first and second layers, respectively. The batchs then will be concatenated
        # into a 7x10000 (size of dataset) matrix from which we can calculate the correlation
        # matrix.

        # The correlation matrix will be of size 7x7, where the first 3x3 block is the correlation
        # matrix for the first layer, the second 4x4 block is the correlation matrix for the second
        # layer, and the 3x4 and 4x3 blocks are the cross-correlation matrices between the first and
        # second layers. The diagonal blocks of the correlation matrix will be the identity matrix
        # since the correlation between a neuron and itself is 1.
        
        # The correlation matrix will be symmetric, so we only need to calculate the upper or lower
        # triangular part of the matrix. We can then binarize the correlation matrix by setting
        # a threshold for the correlation value. If the correlation is above the threshold, then
        # we set the value to 1, otherwise we set the value to 0.
            
        features = [np.concatenate(list(zip(*features))[i]) for i in range(len(features[0]))]
        features = signal_concat(features).T # put in data x features format; samples are rows, features are columns
        
        m, n = features.shape
        print(f"Features size: {(m, n)}")

        if reduction is not None:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.set_float32_matmul_precision('medium') # 'medium' or 'high' for TPU core utilization

            if reduction.__eq__('pca'):
                print(f"Performing PCA on features with {n} components...\n")
                features = self.perform_pca(features, m, alpha=.01, device_list=device_list)
            elif reduction.__eq__('umap'):
                print(f"Performing UMAP on features with {n} components...\n")
                features = self.perform_umap(features, num_components=int(.4*n), num_neighbors=50, min_dist=.175, num_epochs=50, metric='correlation', device_list=device_list)
            else:
                raise ValueError(f"Reduction {reduction} not supported!")

            print(f"Features size after {reduction}: {features.shape}")

            del m

        return features.T, n # put in features x data format; features are rows, samples are columns

    @torch.no_grad()
    def perform_pca(self, features, m, alpha=.05, center_only=True, device_list=None):
        ''' Perform a torch implemented GPU accelerated PCA on the features
            and return the reduced unnormalized features. Expected input shape 
            is (samples, features).
        '''
        features = torch.tensor(features, requires_grad=False).detach().to(device_list[-1])

        # Center the features or normalize them
        if center_only:
            centered = features - features.mean(dim=0)
        else: 
            centered = (features - features.mean(dim=0)) / features.std(dim=0)

        # Perform PCA
        _, S, V = torch.linalg.svd( (centered.T @ centered) / (m - 1), driver='gesvd')
        
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
        features = torch.mm(features, V[:, :num_components]).detach()

        # free up memory on the GPU
        del explained, S, V, num_components, partial_perc
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return features.cpu().data.numpy().astype(np.float64)
    
    @torch.no_grad()
    def perform_umap(self, features, num_components, num_neighbors=50, min_dist=0.1, num_epochs=10, metric='euclidean', device_list=None):
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
            num_workers=os.cpu_count() // 2,
            num_gpus=len(device_list) if device_list is not None else 0,
            match_nonparametric_umap=False # Train network to match embeddings from non parametric umap
        )

        pumap.fit(features)
        features = pumap.transform(features)

        return features
   
    def get_structure(self):
        ''' Collect structure (weights) from the self.network.module.forward_weights() routine '''
        # modified #
        ## NOTICE: only weights are maintained and combined into two dimensions, biases are ignored
        weights = []
        
        [print("we get data type is {}, size is {}".format(type(f.data),f.size())) for f in self.network.parameters()]
        
        for index, var in enumerate(self.network.parameters()):
            if index % 2 == 0:
                f = var.cpu().data.numpy().astype(np.float16) # var as Variable, type(var.data) is Tensor, should be transformed from cuda to cpu(),with type float16
                
                weight = np.reshape(f, (f.shape[0], np.prod(f.shape[1:])))
                print("weight size ==== ", weight.shape)
                
                weights.append(weight)
       
        return weights

''' dask-cuML PCA implementation
    def perform_pca(self, features, client, n_components, alpha=.05):
        from cuml.dask.decomposition import PCA
        import dask_cudf
        # import dask.array as da
        import cudf

        pca = PCA(client=client, n_components=n_components, svd_solver='jacobi', whiten=False, random_state=SEED)
        pca.fit(features)

        Vt = cudf.DataFrame(pca.components_.values)
        # Vt = da.from_array(pca.components_.values)
        # Vt = dask_cudf.from_cudf(Vt, npartitions=1)
        exp_var = pca.explained_variance_ratio_.values

        # print(f'PCA comp type: {type(pca.components_.values)}\n')
        # print(f'Vt type: {type(Vt)}\n')
        # print(f'Explained variance type: {type(exp_var)}\n')
        # print(f'features type: {type(features.to_dask_array())}\n')
        # exit()
        
        comps = 0
        var = 0
        for val in exp_var:
            var += val
            comps += 1
            if var >= 1 - alpha:
                break
        
        print(f'Explained variance: {var:.3f} with {comps} components\n')
        
        del exp_var, var, pca

        # print(f'Vt type: {type(Vt)}\n')
        # print(f'features type: {type(features)}\n')
        # print(f'Vt shape: {Vt.iloc[:,:comps].compute().shape}\n')
        # print(f'features shape: {features.compute().shape}\n')
        # exit()
        # features, Vt = features.align(Vt.iloc[:,:comps], join='right', axis=1)
        
        # features = features.to_dask_array()
        # print(f'Features.value type: {type(features.values)}\n')
        features = cudf.DataFrame(features).dot(Vt[:,:comps]).compute()

        del Vt, comps

        return features
'''
