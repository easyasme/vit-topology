import os

import numpy as np
import torch

from config import SEED
from graph import signal_concat
from utils import progress_bar
from reductions import perform_pca, perform_kmeans, perform_umap


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
    def get_function(self, reduction=None, device_list=None, corr='pearson', exp=1):
        ''' Collect function (features) from the self.network.module.forward_features() routine '''
        features = []

        for batch_idx, (inputs, targets) in enumerate(self.loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            assert not torch.isnan(inputs).any(), 'NaN in inputs at passers.py:get_function()'
            
            # for f in self.network.forward_features(inputs):
            #     assert not torch.isnan(f).any(), 'NaN in forward_features at passers.py:get_function()'

            # Collect activations from each batch - store in features
            activations = self.network.forward_features(inputs)
            assert all(not torch.isnan(f).any() for f in activations), 'NaN in activations at passers.py:get_function()'

            # Convert tensors to numpy arrays
            features.append([f.cpu().data.numpy().astype(np.float32) for f in activations])
                
            progress_bar(batch_idx, len(self.loader))
            
        features = [np.concatenate(list(zip(*features))[i]) for i in range(len(features[0]))]
        features = signal_concat(features).T # put in data x features format; samples are rows, features are columns
        
        m, n = features.shape
        print(f"\nFeatures size: {(m, n)}")

        if reduction is not None:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.set_float32_matmul_precision('medium') # 'medium' or 'high' for TPU core utilization

            if reduction.__eq__('pca'):
                print(f"Performing PCA on features with {n} components...\n")
                # features = self.perform_pca(features, m, alpha=.01, device_list=device_list)
                features = perform_pca(features, m, alpha=.01, device_list=device_list)
            elif reduction.__eq__('umap'):
                print(f"Performing UMAP on features with {n} components...\n")
                # features = self.perform_umap(features, num_components=int(.4*n), num_neighbors=50, min_dist=.175, num_epochs=50, metric='correlation', device_list=device_list)
                features = perform_umap(features, num_components=int(.4*n), num_neighbors=50, min_dist=.175, num_epochs=50, metric='correlation', device_list=device_list)
            elif reduction.__eq__('kmeans'):
                print(f"Performing K-means on features with {n} components...\n")
                # features = self.perform_kmeans(features, device_list=device_list, exp=exp, corr=corr)
                features = perform_kmeans(features, device_list=device_list, exp=exp, corr=corr)
            else:
                raise ValueError(f"Reduction {reduction} not supported!")

            print(f"Features size after {reduction}: {features.shape}")

        del m, n

        return features.T # put in features x data format; features are rows, samples are columns
   
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
