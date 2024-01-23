from __future__ import print_function

import pickle
import time

import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from gph import ripser_parallel
from gtda.homology._utils import _postprocess_diagrams
from gtda.plotting import plot_diagram
from matplotlib import cm
from scipy.sparse import coo_matrix

from bettis import betti_nums
from config import UPPER_DIM
from graph import *
from loaders import *
from models.utils import get_model
from passers import Passer
from utils import *

parser = argparse.ArgumentParser(description='Build Graph and Compute Betti Numbers')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--save_dir')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])
parser.add_argument('--input_size', default=32, type=int)
parser.add_argument('--thresholds', default='0.5 1.0', help='Defining thresholds range in the form \'start stop\' ')
parser.add_argument('--eps_thresh', default=1., type=float)
parser.add_argument('--iter', default=0, type=int)
parser.add_argument('--verbose', default=0, type=int)

args = parser.parse_args()
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

print("Device:", device, "\n")

START_LAYER = 3 if args.net in ['vgg', 'resnet'] else 0

SAVE_DIR = args.save_dir
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

IMG_DIR = os.path.join(SAVE_DIR, 'images')
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

THREED_IMG_DIR = os.path.join(IMG_DIR, '3d_images')
if not os.path.exists(THREED_IMG_DIR):
    os.makedirs(THREED_IMG_DIR)

PERS_DIR = os.path.join(IMG_DIR, 'persistence')
if not os.path.exists(PERS_DIR):
    os.makedirs(PERS_DIR)

CURVES_DIR = os.path.join(IMG_DIR, 'curves')
if not os.path.exists(CURVES_DIR):
    os.makedirs(CURVES_DIR)

# Build models
print('==> Building model..')
net = get_model(args.net, args.dataset)
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('\n', net, '\n')

''' Prepare criterion '''
criterion = nn.CrossEntropyLoss()

''' Prepare val data loader '''
functloader = loader(args.dataset+'_test', batch_size=100, iter=args.iter, subset=list(range(0, 1000)), verbose=False)

start = float(args.thresholds.split(' ')[0])
stop = float(args.thresholds.split(' ')[1])
thresholds = np.linspace(start=start, stop=stop, num=10)

eps_thresh = np.linspace(start=0., stop=args.eps_thresh, num=10)

''' Load checkpoint and get activations '''
for epoch in vars(args)['chkpt_epochs']:
    print('\n'+'==> Loading checkpoint for epoch {}...'.format(epoch)+'\n')
    
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    
    checkpoint = torch.load('./checkpoint/' + args.net + '/' + args.net + '_' + args.dataset + '_ss' + str(args.iter) + '/' +'ckpt_epoch_' + str(epoch)+'.pt', map_location=device)
    
    net.load_state_dict(checkpoint['net'])

    ''' Define passer and get activations '''
    passer = Passer(net, functloader, criterion, device)
    activs = passer.get_function()
    activs = signal_concat(activs)
    adj = adjacency(activs)

    num_nodes = adj.shape[0] if adj.shape[0] != 0 else 1

    if args.verbose:
        print('\n', 'The dimension of the adjacency matrix is {}'.format(adj.shape))
        print('Adj mean {}, min {}, max {}'.format(np.mean(adj), np.min(adj), np.max(adj)), '\n')

    betti_nums_list = []
    betti_nums_list_3d = []
    for j, t in enumerate(thresholds):
        print('\n', 'Epoch:', epoch, '| Threshold:', t, '\n')

        binadj = partial_binarize(np.copy(adj), t) # partially binarize adj matrix
        
        flip = make_flip_matrix(binadj) # make flip matrix
        binadj = flip - binadj # flip adj matrix
        
        np.fill_diagonal(binadj, 0.) # remove self-loops for distance matrix format
        binadj = coo_matrix(binadj) # convert to sparse matrix

        if args.verbose:
            print('\n', 'The dimension of the COO adjacency matrix is {}'.format(binadj.shape))
            print('Binadj mean {}, min {}, max {}'.format(np.mean(binadj), np.min(binadj), np.max(binadj)))

        comp_time = time.time()
        dgm = ripser_parallel(binadj, metric="precomputed", maxdim=UPPER_DIM, n_threads=-1, collapse_edges=True)
        comp_time = time.time() - comp_time
        
        print(f'\n Computation time: {comp_time/60:.3f} minutes \n')

        dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", range(UPPER_DIM + 1), np.inf, True)[0]

        betti_nums_3d = []
        for k in eps_thresh:
            betti_nums_3d.append(betti_nums(dgm_gtda, thresh=k))
        betti_nums_list_3d.append(betti_nums_3d)
        
        betti_nums_list.append(betti_nums(dgm_gtda, thresh=args.eps_thresh))

        if (t != 1.0) and (j % 25 == 0):
            dgm_img_path = PERS_DIR + "/epoch_{}_thresh{:.2f}_b{}".format(epoch, t, UPPER_DIM) + ".png"

            dgm_fig = plot_diagram(dgm_gtda)
            dgm_fig.write_image(dgm_img_path)

    betti_nums_list = np.array(betti_nums_list)
    betti_nums_list_3d = np.array(betti_nums_list_3d)
    
    # Plot betti numbers per dimension at current eps. thresh.
    for i in range(0, UPPER_DIM+1):
        bn_img_path = CURVES_DIR + "/epoch_{}_dim_{}_bn_{}".format(epoch, UPPER_DIM, i) + ".png"
        
        fig = plt.figure()
        
        color = 'b' if i == 1 else 'r' if i == 2 else 'g' if i == 3 else 'y'

        plt.plot(thresholds, betti_nums_list[:, i] / num_nodes, label='Betti {}'.format(i), color=color)

        max_idx = np.argmax(betti_nums_list[:, i] / num_nodes)
        max_val = betti_nums_list[max_idx, i] / num_nodes
        plt.vlines(x=thresholds[max_idx], ymin=0, ymax=max_val, color='orange', linestyle='dashed', label='Max loc. {:.3f}'.format(thresholds[max_idx]))
        plt.hlines(y=max_val, xmin=start, xmax=stop, color='orange', linestyle='dashed', label='Max val. {:.3f}'.format(max_val))

        plt.xlabel('Thresholds')
        plt.ylabel('Betti Numbers')
        plt.ylim(0, 1.1 * max_val)
        plt.grid()
        plt.title(f"Epoch {epoch}")
        plt.legend()
        
        fig.savefig(bn_img_path)
        
        plt.close(fig)

    # 3D plots of betti numbers per dimension
    for i in range(0, UPPER_DIM+1):
        bn3d_img_path = THREED_IMG_DIR + "/epoch_{}_dim_{}_bn_{}_3d".format(epoch, UPPER_DIM, i) + ".pkl"

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        X, Y = np.meshgrid(eps_thresh, thresholds)
        Z = betti_nums_list_3d[:,:,i] / num_nodes
        ax.plot_surface(X, Y, Z, cmap=cm.Spectral, alpha=0.5)

        ax.set_xlabel('Eps. Thresholds')
        ax.set_ylabel('Thresholds')
        ax.set_zlabel('Betti Numbers per Node')
        ax.set_label(f"Betti {i}")
        ax.set_title(f"Epoch {epoch}")

        with open(bn3d_img_path, 'wb') as f:
            pickle.dump(fig, f)

        plt.close(fig)
