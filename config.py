import numpy as np

# Where to save topology results
SAVE_PATH = 'results'

# DIPHA
DIPHA_MAGIC_NUMBER = 8067171840
ID = 7

# Sets upper limit for epsilon range in sparse distance matrix computation
MAX_EPSILON = 0.3

# Number of cores MPI can use
NPROC = 8

# Sets upper limit to dimension to compute for persistent homology.
UPPER_DIM = 2

# Create 30 random lists of 10 class labels each for subsetting imagenet data
np.random.seed(1234)
SUBSETS_LIST = [np.random.randint(0, 1000, size=10) for _ in range(30)]

# print(np.unique(SUBSETS_LIST, axis=0))