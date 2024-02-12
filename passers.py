from utils import progress_bar
import torch
import numpy as np


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
                progress_bar((r-1)*len(self.loader)+batch_idx, r*len(self.loader), 'repeat %d -- Mean Loss: %.3f | Last Loss: %.3f | Acc: %.3f%%'
                             % (r, np.mean(losses), losses[-1], np.mean(accuracies)))

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
    def get_function(self, forward='selected'):
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

            if forward=='selected':
                features.append([f.cpu().data.numpy().astype(np.float64) for f in self.network.forward_features(inputs)])
            elif forward=='parametric':
                features.append([f.cpu().data.numpy().astype(np.float64) for f in self.network.forward_param_features(inputs)])
                
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

        return [np.concatenate(list(zip(*features))[i]) for i in range(len(features[0]))]

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
