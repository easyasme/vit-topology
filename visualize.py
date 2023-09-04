import matplotlib.pyplot as plt
import scipy.ndimage
from bettis import *
import argparse
import os
from config import SAVE_PATH, MAX_EPSILON


parser = argparse.ArgumentParser()
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0)
parser.add_argument('--epochs', nargs='+', type=int)
parser.add_argument('--dim', default=1, type=int)
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()

directory = os.path.join(SAVE_PATH, args.net + '_' + args.dataset + '_' + 'ss' + str(args.iter))
bin_dir = os.path.join(directory, 'bin/')

trial = 0

EMPTY = 0

for epc in args.epochs:
    #  read persistent diagram from persistent homology output
    birth, death = read_results(bin_dir, epc, trl=args.trial, max_epsilon=MAX_EPSILON, dim=args.dim, persistence=0.02)

    if len(birth) > 0:
        #  compute betti curve from persistent diagram
        x, betti = pd2betti(birth, death)

        #  filter curve for improved visualization
        filter_size = int(len(betti) / 10)
        betti = scipy.ndimage.uniform_filter1d(betti, size=filter_size, mode='constant')

        # plot curve
        plt.xlabel('$\epsilon$')
        plt.ylabel('Number of Cavities (N)')
        plt.plot(x, betti, label='Epoch ' + str(epc))
        plt.legend()
        plt.title('Betti ' + "{}".format(args.dim))

        # compute life and midlife
        life = pd2life(birth, death)
        midlife = pd2midlife(birth, death)
        print('EPC = {}, LIFE = {}, MIDLIFE = {}'.format(epc, life, midlife))
    else:
        EMPTY += 1
        print('The persistence diagram is empty!')

# plt.show()

if not os.path.exists(directory + "/images"):
    os.makedirs(directory + "/images")

if not EMPTY == len(args.epochs):
    plt.savefig(directory + "/images/" + args.net + "_" + args.dataset + "_" + "ss" + str(args.iter) + "_epoch_" + str(epc) + "_b" + str(args.dim) + ".png")