import numpy as np


def betti_nums(pd, thresh=0.):
    ''' This function assumes that the persistence diagram is in giotto-ph format;
    thresh: threshold for computing betti nums.'''

    max_dim = int(np.max(pd[:, 2]))
    betti_nums = np.zeros((max_dim+1,))

    dct = {}
    for dim in range(max_dim+1):
        dct[dim] = []
    for x in pd:
        dct[int(x[-1])].append(tuple(x[:-1]))

    for dim in range(max_dim+1):
        for (b, d) in dct[dim]:
            if (d - b > thresh - b) and (thresh - b >= 0.):
                betti_nums[dim] += 1
    
    return betti_nums

def pd2life(birth, death):
    return np.mean(np.asarray(death) - np.asarray(birth))

def pd2midlife(birth, death):
    return np.mean(np.asarray(death) + np.asarray(birth) / 2)

def pd2mullife(birth, death):
    return np.mean(np.asarray(death) / np.asarray(birth))

def betti_integral(bettis, t_begin, t_end, delta=0.05):
    """ Compute integral """
    return np.sum([bettis['t_{:1.2f}'.format(x)] for x in np.arange(t_begin, t_end, delta)], axis=0)
