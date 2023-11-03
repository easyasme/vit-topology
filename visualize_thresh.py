import matplotlib.pyplot as plt
import scipy.ndimage
from bettis import *
from graph import *
import argparse
import os
from config import SAVE_PATH, MAX_EPSILON, UPPER_DIM


parser = argparse.ArgumentParser()
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0)
parser.add_argument('--epochs', nargs='+', action='extend', type=list, default=[1, 5, 10, 20])
parser.add_argument('--dim', default=1, type=int)
parser.add_argument('--thresholds', default='0.5 1.0', help='Defining thresholds range in the form \'start stop\' ')
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()

directory = os.path.join(SAVE_PATH, args.net + '/' +  args.net + '_' + args.dataset + '_' + 'ss' + str(args.iter))
bin_dir = os.path.join(directory, 'bin/')

if not os.path.exists(directory + "/images"):
    os.makedirs(directory + "/images")

img_path = directory + "/images/" + args.net + "_" + args.dataset + "_" + "ss" + str(args.iter) + "_epoch_" + str(args.epochs[-1]) + "_b" + str(args.dim) + "_thresh" + ".png"

EMPTY = 0

start = float(args.thresholds.split(' ')[0])
stop = float(args.thresholds.split(' ')[1])

thresholds = np.linspace(start=start, stop=stop, num=10)
curves = []

for epc in args.epochs:
    for t in thresholds:
        # if not os.path.exists(img_path):
            #  read persistent diagram from persistent homology output

        out_file = bin_dir + 'adj_epc{}_thresh{:.2f}_{}_dim{}.bin.out'.format(epc, t, MAX_EPSILON, UPPER_DIM)

        if os.path.exists(out_file):
            print('Reading results for subset {} epoch {}'.format(args.iter, epc) + '\n')
            birth, death = read_results(out_file, dim=args.dim, persistence=0.02)
        else:
            continue

        if len(birth) > 0:
            #  compute betti curve from persistent diagram
            _, betti = pd2betti(birth, death)

            #  filter curve for improved visualization
            filter_size = int(len(betti) / 10)

            # sliding window average of size filter_size
            # where 'constant' means that the input is 
            # extended by filling all values beyond the 
            # edge with the same constant value, i.e. 0.0
            betti = scipy.ndimage.uniform_filter1d(betti, size=filter_size, mode='constant')
            print('Betti curve: ', betti)
            print('Betti sum: ', betti.sum())
            curves.append(betti.sum())

            # compute life and midlife
            life = pd2life(birth, death)
            midlife = pd2midlife(birth, death)

            print('EPC = {}, LIFE = {}, MIDLIFE = {}'.format(epc, life, midlife))
        else:
            EMPTY += 1
            print('The persistence diagram is empty! \n')

    # plot curve
    plt.xlabel('$\T$')
    plt.ylabel('Number of Cavities')
    plt.plot(thresholds, curves, label='Epoch ' + str(epc))
    plt.legend()
    plt.title('Epoch {}'.format(epc))

    if not EMPTY == len(args.epochs):
        plt.savefig(img_path)