import os
import argparse
import numpy as np

from config import NPROC, MAX_EPSILON, UPPER_DIM


parser = argparse.ArgumentParser()

parser.add_argument('--save_path')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--epochs', nargs='+', action='extend', type=list, default=[1, 5, 10, 20])
parser.add_argument('--thresholds', default='0.5 1.0', help='Defining thresholds range in the form \'start stop\' ')
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()

path = os.path.join(args.save_path, args.net + "_" + args.dataset + '_' + 'ss' + str(args.iter))
path = os.path.join(path, "bin/")

start = float(args.thresholds.split(' ')[0])
stop = float(args.thresholds.split(' ')[1])

for e in args.epochs:
    for t in np.linspace(start=start, stop=stop, num=10):
        if not os.path.exists(path + "adj_epc{}_thresh{:.2f}_{}.bin".format(e, t, MAX_EPSILON)):
            os.system("../dipha/build/full_to_sparse_distance_matrix " + str(MAX_EPSILON) + " " + path + "adj_epc{}_thresh{:.2f}.bin ".format(e, t) + path + "adj_epc{}_thresh{:.2f}_{}.bin".format(e, t, MAX_EPSILON))

        if not os.path.exists(path + "adj_epc{}_thresh{:.2f}_{}_dim{}.bin.out".format(e, t, MAX_EPSILON, UPPER_DIM)):
            os.system("mpiexec -n " + str(NPROC) + " .././dipha/build/dipha --upper_dim " + str(UPPER_DIM) + " --benchmark  --dual " + path + "adj_epc{}_thresh{:.2f}_{}.bin ".format(e, t, MAX_EPSILON) + path + "adj_epc{}_thresh{:.2f}_{}_dim{}.bin.out".format(e, t, MAX_EPSILON, UPPER_DIM))