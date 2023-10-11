import os
import argparse

# from torch.cuda import is_available
from config import NPROC, MAX_EPSILON, UPPER_DIM


parser = argparse.ArgumentParser()

parser.add_argument('--save_path')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0)
parser.add_argument('--epochs', nargs='+')
parser.add_argument('--thresholds', nargs='+', type=float)
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()

path = os.path.join(args.save_path, args.net + "_" + args.dataset + '_' + 'ss' + str(args.iter))
path = os.path.join(path, "bin/")

# gpu = is_available()

for e in args.epochs:
    if not os.path.exists(path + "adj_epc{}_trl{}_{}.bin".format(e, args.trial, MAX_EPSILON)):
        os.system("../dipha/build/full_to_sparse_distance_matrix " + str(MAX_EPSILON) + " " + path + "adj_epc{}_trl{}.bin ".format(e, args.trial) + path + "adj_epc{}_trl{}_{}.bin".format(e, args.trial, MAX_EPSILON))

    # if gpu:
    #     os.system("mpiexec --mca opal_cuda_support 1 -n " + str(NPROC) + " .././dipha/build/dipha --upper_dim " + str(UPPER_DIM) + " --benchmark  --dual " + path + "adj_epc{}_trl{}_{}.bin ".format(e, args.trial, MAX_EPSILON) + path + "adj_epc{}_trl{}_{}.bin.out".format(e, args.trial, MAX_EPSILON))
    # else:
    
    # inputfile = path + "adj_epc{}_trl{}_{}.bin".format(e, args.trial, MAX_EPSILON)
    # rip_dict = os.system("mpiexec -n " + str(NPROC) + " .././ripser++ --format dipha --sparse --dim " + str(UPPER_DIM) + inputfile)

    # print(rip_dict)

    # if not os.path.exists(path + "adj_epc{}_trl{}_{}.bin.out".format(e, args.trial, MAX_EPSILON)):
    os.system("mpiexec -n " + str(NPROC) + " .././dipha/build/dipha --upper_dim " + str(UPPER_DIM) + " --benchmark  --dual " + path + "adj_epc{}_trl{}_{}.bin ".format(e, args.trial, MAX_EPSILON) + path + "adj_epc{}_trl{}_{}.bin.out".format(e, args.trial, MAX_EPSILON))
