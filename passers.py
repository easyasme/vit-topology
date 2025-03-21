import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import progress_bar, get_accuracy, create_img_from_tensors
from reductions import *


class Passer():
    def __init__(self, net, loader, criterion, device, train=False, val=False, repeat=1):
        self.network = net
        self.training = train
        self.validation = val
        self.criterion = criterion
        self.device = device
        self.loader = loader
        self.repeat = repeat

    def _pass(self, optimizer=None, mask=None):
        ''' Main data passing routing '''
        losses = []
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

    def get_function(self, reduction=None, device_list=None, corr='pearson', exp=1, average=False, save_imgs=False):
        ''' Collect function (features) from the self.network.forward_features() routine '''
        features = []

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.loader):
                inputs = inputs.to(self.device)

                # Collect activations from each batch - store in features
                activations = self.network.forward_features(inputs)
                activations = torch.stack(activations)

                features.append(activations)
                    
                progress_bar(batch_idx, len(self.loader))
        features = torch.cat(features, dim=1) # layers x samples x num_vec x emb_dim

        if average:
            features = torch.mean(features, dim=1)

        if save_imgs:
            create_img_from_tensors(features)

        print(f"\nFeatures size before reduction: {features.size()}")

        if reduction is not None:
            torch.set_float32_matmul_precision('medium') # 'medium' or 'high' for TPU core utilization

            if reduction.__eq__('pca'):
                print(f"Performing PCA...\n")
                features = perform_pca(features, features.size()[-2], alpha=.01, device_list=device_list)
            elif reduction.__eq__('umap'):
                print(f"Performing UMAP...\n")
                features = perform_umap(features,
                                        num_components=int(.4*features.size()[-1]),
                                        num_neighbors=50,
                                        min_dist=.175,
                                        num_epochs=50,
                                        metric='correlation',
                                        device_list=device_list)
            elif reduction.__eq__('kmeans'):
                print(f"Performing K-means...\n")
                features = perform_kmeans(features, device_list=device_list, exp=exp, corr=corr)
            elif reduction.__eq__('cla'): # return list of tensors for each encoder block; reduces vectors
                print(f"Performing CLA...\n")
                if len(features.size()) == 4:
                    for layer in torch.unbind(features, dim=0):
                        layers = []
                        print(f"Layer size: {layer.shape}")
                        red_features = []
                        for i,sample in enumerate(torch.unbind(layer, dim=0)):

                            print(f"Sample number: {i}, Sample size: {sample.shape}")
                            red_features.append(perform_cla(sample, device_list=device_list)[0])
                        layers.append(torch.stack(red_features))
                    features = layers
                else:
                    red_features = []
                    for activation in torch.unbind(features, dim=0):
                        red_features.append(perform_cla(activation, device_list=device_list)[0])
                    features = red_features
            else:
                raise ValueError(f"Reduction {reduction} not supported!")
        else:
            features = torch.unbind(features, dim=0) # list of tensors for each encoder block
        
        print(f"\nFeatures size after reduction: {len(features), features[0].shape}")

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        return features # list of tensors for each encoder block; reduces vectors if reduction is not None

    def get_predictions(self):
        ''' Returns predictions and targets '''
        preds, gts = [], []

        for batch_idx, (inputs, targets) in enumerate(self.loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.network(inputs)

            gts.append(targets.cpu().data.numpy())
            preds.append(outputs.cpu().data.numpy().argmax(1))
            
        return np.concatenate(gts), np.concatenate(preds)

    def get_sample(self):
        iterator = iter(self.loader)
        inputs, _ = iterator.next()

        return inputs[0:1,...].to(self.device)

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

    def run(self, optimizer=None, mask=None):
        if optimizer:
            self.network.train()

            return self._pass(optimizer, mask=mask)
        else:
            self.network.eval()

            with torch.no_grad():
                return self._pass(mask=mask)
