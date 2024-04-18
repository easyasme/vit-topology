from math import floor

import numpy as np
import pandas as pd
import torch
from tqdm import trange
from scipy.spatial import distance_matrix

def adjacency_l2(signals):
    ''' In this case signal is an 1XN array not a time series. 
    Builds adjacency based on L2 norm between node activations.
    '''
    x = np.tile(signals, (signals.size, 1))
    return np.sqrt((signals - signals.transpose())**2)

@torch.no_grad()
def spearman_ranks(signals, device='cpu'):
    ''' In this case signals is an MXN tensor not a time series. 
    Builds adjacency based on Spearman correlation between node activations.
    '''
    signals = [pd.Series(signals.numpy(force=True)[i]).rank().values for i in range(signals.size(0))]
    signals = torch.tensor(np.array(signals), device=device).detach()
    
    return signals

@torch.no_grad()
def vec_diff(x, device='cpu'):
    ''' Calculate the distance between the components of two 1D tensors. '''
    return torch.cdist(x.view(-1, 1), x.view(-1, 1)).to(device)

@torch.no_grad()
def dist_corr(signals, device):
    ''' In this case signals is an MXN tensor not a time series. 
    Builds adjacency based on distance correlation between node activations.
    '''
    n, _ = signals.size()
    adj = torch.zeros((n, n)).detach()

    print('\nComputing differences...')
    diffs = [vec_diff(signals[i,:], device=device).detach() for i in trange(n, leave=False)]
    print('\nComputing centers...')
    centers = [diffs[i]-diffs[i].mean(dim=0, keepdim=True)-diffs[i].mean(dim=1, keepdim=True)+diffs[i].mean() for i in trange(n, leave=False)]

    del diffs
    torch.cuda.empty_cache()

    print('\nComputing distance correlation...')
    for i in trange(n, leave=False):
        for j in range(i+1, n):
            dCov = torch.sum(centers[i]*centers[j])
            dVar1 = torch.sum(centers[i]*centers[i])
            dVar2 = torch.sum(centers[j]*centers[j])
            
            # skip overall sqrt so that the correlation can be used as a distance metric
            adj[i,j] = dCov/torch.sqrt(dVar1*dVar2) if (dVar1*dVar2 > 0) else 0
        centers[i].cpu()

    adj += adj.clone().T
    adj.fill_diagonal_(1)

    del n, centers, dCov, dVar1, dVar2
    torch.cuda.empty_cache()

    assert torch.all(adj == adj.T), 'Adjacency matrix is not symmetric'
    assert torch.all(torch.diag(adj) == 1), 'Adjacency matrix diagonal is not 1'
    assert torch.all(adj >= 0), 'Adjacency matrix has negative values'

    return adj

@torch.no_grad()
def partial_binarize(M, binarize_t, device):
    ''' Binarize matrix. Real subunitary values. '''
    
    M[M<=binarize_t] = 0

    return M.to(device)

@torch.no_grad()
def adjacency(signals, device, metric=None):
    '''
    Build matrix A of dimensions nxn where a_{ij} = metric(a_i, a_j).
    signals: nxm matrix where each row (signal[k], k=range(n)) is a signal. 
    metric: a function f(.,.) that takes two 2D ndarrays and outputs a single real number (e.g correlation, KL divergence etc).
    '''
    
    signals = np.reshape(signals, (signals.shape[0], -1))
    signals = torch.tensor(signals, device=device, dtype=torch.float32).detach()
        
    if metric == 'spearman':
        signals = spearman_ranks(signals, device=device)
        adj = torch.nan_to_num(torch.corrcoef(signals)).detach()
    elif metric == 'dcorr':
        adj = dist_corr(signals, device=device).detach()
    elif callable(metric):
        n, _ = signals.shape

        adj= [[metric(signals[i], torch.transpose(signals[j])) for j in range(n)] for i in range(n)]

        ''' Normalize '''
        adj = robust_scaler(adj)
        adj = torch.nan_to_num(adj).detach()
    else:
        adj = torch.nan_to_num(torch.corrcoef(signals)).detach()

    del signals
    torch.cuda.empty_cache()

    return adj.to(device)

@torch.no_grad()
def minmax_scaler(A):
    A = (A - A.min())/A.max()
    return A

@torch.no_grad()
def standard_scaler(A):
    return  np.abs((A - np.mean(A))/np.std(A))

@torch.no_grad()
def robust_scaler(A, quantiles=[0.05, 0.95]):
    a = np.quantile(A, quantiles[0])
    b = np.quantile(A, quantiles[1])
    
    return (A-a)/(b-a)

def signal_concat(signals):
    return np.concatenate([np.transpose(x.reshape(x.shape[0], -1)) for x in signals], axis=0)

def build_density_adjacency(adj, density_t):
    ''' Binarize matrix '''
    total_edges = np.prod(adj.shape)
    t, t_decr = 1, 0.001
    while True:
        ''' Decrease threshold until density is met '''
        edges = np.sum(adj > t)
        density = edges/total_edges
        '''print('Threshold: {}; Density:{}'.format(t, density))'''
        
        if density > density_t:
            break

        t = t-t_decr
        
    return t
