import numpy as np
import torch
import time
import os

from graph import signal_concat
from utils import progress_bar
from config import SEED


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
    def get_function(self, num_devs=-1, reduction=None, cluster=None, device_list=None):
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

        if reduction is not None:
            # import cudf
            # import dask_cudf
            # from dask.distributed import Client

            m, n = features.shape
            print(f"Features size before {reduction}: {(m, n)}")

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # features must be in dask cuDF format for distributed computing
            # features = cudf.DataFrame(features)
            # if num_devs != -1:
            #     features = dask_cudf.from_cudf(features, npartitions=3* num_devs)
            # else:
            #     features = dask_cudf.from_cudf(features, npartitions=1)

            # # Perform dimensionality reduction
            # client = Client(cluster)
            if reduction.__eq__('pca'):
                features = self.perform_pca(features, m, alpha=.05, device_list=device_list)
                # features = self.perform_pca(features, client=client, n_components=n, alpha=.025)
            elif reduction.__eq__('umap'):
                features = self.perform_umap(features, n_components=n)
            elif reduction.__eq__('tsne'):
                features = self.perform_tsne(features, device_list=device_list)
            # client.close()

            print(f"Features size after {reduction}: {features.shape}")

        return features.T # put in features x data format; features are rows, samples are columns

    @torch.no_grad()
    def perform_pca(self, features, m, alpha=.05, device_list=None):
        ''' Perform PCA on the features '''
        tens_feats = torch.tensor(features, requires_grad=False).to(device_list[-1]).detach() # transpose to get the right shape

        # Normalize the features
        # centered = tens_feats - tens_feats.mean(dim=0)
        tens_feats = (tens_feats - tens_feats.mean(dim=0)) / tens_feats.std(dim=0)

        # Perform PCA
        # cov = (torch.mm(centered.T, centered) / (m - 1)).to(device_list[-2]).type(torch.float32)
        cov = (torch.mm(tens_feats.T, tens_feats) / (m - 1)).to(device_list[-1]).type(torch.float32).detach()
        
        svd_time = time.time()
        _, S, V = torch.linalg.svd(cov, driver='gesvdj', full_matrices=False)
        print(f'SVD time: {time.time() - svd_time:.3f}s')
        
        S, V = S.numpy(force=True).detach(), V.to(device_list[-1]).detach()

        # Calculate the number of principal components to keep
        explained = S / sum(S) # calculate the percentage of variance explained by each component
        
        num_components = 0
        partial_perc = 0
        for perc in explained:
            partial_perc += perc
            num_components += 1
            if partial_perc >= 1 - alpha:
                break

        # Project the data onto the principal components
        proj = torch.mm(tens_feats, V[:, :num_components]).detach()

        del tens_feats, explained, cov, S, V, num_components, partial_perc
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return proj.cpu().data.numpy().astype(np.float64)

    # def perform_pca(self, features, client, n_components, alpha=.05):
    #     from cuml.dask.decomposition import PCA
    #     import dask_cudf
    #     # import dask.array as da
    #     import cudf

    #     pca = PCA(client=client, n_components=n_components, svd_solver='jacobi', whiten=False, random_state=SEED)
    #     pca.fit(features)

    #     Vt = cudf.DataFrame(pca.components_.values)
    #     # Vt = da.from_array(pca.components_.values)
    #     # Vt = dask_cudf.from_cudf(Vt, npartitions=1)
    #     exp_var = pca.explained_variance_ratio_.values

    #     # print(f'PCA comp type: {type(pca.components_.values)}\n')
    #     # print(f'Vt type: {type(Vt)}\n')
    #     # print(f'Explained variance type: {type(exp_var)}\n')
    #     # print(f'features type: {type(features.to_dask_array())}\n')
    #     # exit()
        
    #     comps = 0
    #     var = 0
    #     for val in exp_var:
    #         var += val
    #         comps += 1
    #         if var >= 1 - alpha:
    #             break
        
    #     print(f'Explained variance: {var:.3f} with {comps} components\n')
        
    #     del exp_var, var, pca

    #     # print(f'Vt type: {type(Vt)}\n')
    #     # print(f'features type: {type(features)}\n')
    #     # print(f'Vt shape: {Vt.iloc[:,:comps].compute().shape}\n')
    #     # print(f'features shape: {features.compute().shape}\n')
    #     # exit()
    #     # features, Vt = features.align(Vt.iloc[:,:comps], join='right', axis=1)
        
    #     # features = features.to_dask_array()
    #     # print(f'Features.value type: {type(features.values)}\n')
    #     features = cudf.DataFrame(features).dot(Vt[:,:comps]).compute()

    #     del Vt, comps

    #     return features

    @torch.no_grad()
    def perform_umap(self, features, scale=2):
        ''' Perform UMAP on the features '''
        from cuml.manifold import UMAP
        from cuml.dask.manifold import UMAP as MNMG_UMAP
        import dask.array as da

        local_model = UMAP(n_components=features.shape[1] // scale)
        local_model.fit(features)

        dist_model = MNMG_UMAP(model=local_model)
        dist_feats = da.from_array(features, chunks=(500, -1))
        embedding = dist_model.transform(dist_feats)

        return embedding.compute().to_numpy()
    
    @torch.no_grad()
    def perform_tsne(self, features):
        ''' Perform t-SNE on the features '''
        from sklearn.manifold import TSNE

        return TSNE(n_components=2).fit_transform(features)

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
