import os
import argparse

from config import NPROC, MAX_EPSILON, UPPER_DIM
from torch.cuda import is_available


parser = argparse.ArgumentParser()

parser.add_argument('--save_path')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0)
parser.add_argument('--epochs', nargs='+')
parser.add_argument('--thresholds', nargs='+', type=float)

args = parser.parse_args()

path = os.path.join(args.save_path, args.net + "_" + args.dataset)
path = os.path.join(path, "bin/")

# --mca opal_cuda_support 1 for cuda support

cuda_avail = is_available()

for e in args.epochs:
    os.system("../dipha/build/full_to_sparse_distance_matrix " + str(MAX_EPSILON) + " " + path + "adj_epc{}_trl{}.bin ".format(e, args.trial) + path + "adj_epc{}_trl{}_{}.bin".format(e, args.trial, MAX_EPSILON))
    
    if not cuda_avail:
        os.system("mpiexec.openmpi -n " + str(NPROC) + " --mca opal_cuda_support 1" + " ../dipha/build/dipha --upper_dim " + str(UPPER_DIM) + " --benchmark  --dual " + path + "adj_epc{}_trl{}_{}.bin ".format(e, args.trial, MAX_EPSILON) + path + "adj_epc{}_trl{}_{}.bin.out".format(e, args.trial, MAX_EPSILON))
    else:
        os.system("mpiexec.openmpi -n " + str(NPROC) + " ../dipha/build/dipha --upper_dim " + str(UPPER_DIM) + " --benchmark  --dual " + path + "adj_epc{}_trl{}_{}.bin ".format(e, args.trial, MAX_EPSILON) + path + "adj_epc{}_trl{}_{}.bin.out".format(e, args.trial, MAX_EPSILON))
