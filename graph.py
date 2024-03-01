import numpy as np
import torch


def adjacency_l2(signal):
    ''' In this case signal is an 1XN array not a time series. 
    Builds adjacency based on L2 norm between node activations.
    '''
    x = np.tile(signal, (signal.size, 1))
    return np.sqrt((signal - signal.transpose())**2)

@torch.no_grad()
def partial_binarize(M, binarize_t, device):
    ''' Binarize matrix. Real subunitary values. '''
    
    M[M<=binarize_t] = 0

    return M.to(device)

@torch.no_grad()
def make_flip_matrix(M, device):
    ''' Takes in thresholded distance matrix M and returns a matrix of 1s and 0s
    where non-zero entries in M are 1 in the returned matrix '''

    flipped = (M!=0).type(torch.float32)
    
    return flipped.to(device)

@torch.no_grad()
def adjacency(signals, device, metric=None):
    '''
    Build matrix A of dimensions nxn where a_{ij} = metric(a_i, a_j).
    signals: nxm matrix where each row (signal[k], k=range(n)) is a signal. 
    metric: a function f(.,.) that takes two 1D ndarrays and outputs a single real number (e.g correlation, KL divergence etc).
    '''
    
    ''' Get input dimensions '''
    signals = np.reshape(signals, (signals.shape[0], -1))
    signals = torch.tensor(signals, device=device, dtype=torch.float16).detach()

    ''' If no metric provided fast-compute correlation  '''
    if not metric:
        adj = torch.abs(torch.nan_to_num(torch.corrcoef(signals))).detach()
        del signals
        torch.cuda.empty_cache()
        return adj.to(device)
        
    n, _ = signals.shape

    A = [[metric(signals[i], torch.transpose(signals[j])) for j in range(n)] for i in range(n)]

    ''' Normalize '''
    A = robust_scaler(A)
    
    print("End adjacency...\n")

    return torch.abs(torch.nan_to_num(A)).detach().to(device)

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
