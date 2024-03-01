from __future__ import print_function

import argparse
import time

from gph import ripser_parallel
from gtda.homology._utils import _postprocess_diagrams
from gtda.plotting import plot_diagram
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
parser.add_argument('--cluster', default=None, help='Dask client')
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])
parser.add_argument('--input_size', default=32, type=int)
parser.add_argument('--thresholds', default='.05 1.0', help='Defining thresholds range in the form \'start stop\' ')
parser.add_argument('--eps_thresh', default=1., type=float)
parser.add_argument('--reduction', default='pca', type=str, help='Reductions: pca, tsne, umap, or none.')
parser.add_argument('--iter', default=0, type=int)
parser.add_argument('--verbose', default=0, type=int)

args = parser.parse_args()

device_list = []
NUM_DEVICES = torch.cuda.device_count()
if NUM_DEVICES > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    device_list = [torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count())]
else:
    print("Using a single GPU")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_list.append(device)

for i, device in enumerate(device_list):
    print(f"Device {i}: {device}")

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
print('\n ==> Building model..')
net = get_model(args.net, args.dataset)
net = net.to(device_list[0])

''' Prepare criterion '''
criterion = nn.CrossEntropyLoss()

''' Prepare val data loader '''
functloader = loader(args.dataset+'_test', batch_size=100, iter=args.iter, subset=list(range(0, 1000)), verbose=False)

start = float(args.thresholds.split(' ')[0])
stop = float(args.thresholds.split(' ')[1])
thresholds = np.linspace(start=start, stop=stop, num=50, dtype=np.float64)

eps_thresh = np.linspace(start=0., stop=args.eps_thresh, num=50)

total_time = 0.
''' Load checkpoint and get activations '''
with torch.no_grad():
    for epoch in vars(args)['chkpt_epochs']:
        print(f'\n==> Loading checkpoint for epoch {epoch}...\n')
        
        assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
        
        checkpoint = torch.load(f'./checkpoint/{args.net}/{args.net}_{args.dataset}_ss{args.iter}/ckpt_epoch_{epoch}.pt', map_location=device_list[0])
        
        net.load_state_dict(checkpoint['net'])
        net.eval()

        ''' Define passer and get activations '''
        passer = Passer(net, functloader, criterion, device_list[0])
        activs = passer.get_function(reduction=args.reduction, cluster=args.cluster, num_devs=NUM_DEVICES, device_list=device_list)
        adj = adjacency(activs, device=device_list[0])

        num_nodes = adj.shape[0] if adj.shape[0] != 0 else 1

        if args.verbose:
            print(f'\n The dimension of the corrcoef matrix is {adj.size()[0], adj.size()[-1]} \n')
            print(f'Adj mean {adj.mean():.4f}, min {adj.min():.4f}, max {adj.max():.4f} \n')

        betti_nums_list = []
        betti_nums_list_3d = []
        for j, t in enumerate(thresholds):
            print(f'\n Epoch: {epoch} | Threshold: {t:.3f} \n')

            # print(f'Activs dtype {activs.dtype}')
            # print(f'Adj dtype {adj.dtype}')
            
            binadj = partial_binarize(adj.detach().clone(), t, device=device_list[-1])
            flip = make_flip_matrix(binadj.detach(), device=device_list[-1])
            
            # print(f'Binadj dtype {binadj.dtype}, flip dtype {flip.dtype}')

            binadj *= -1 # flip binadj to reflect closeness
            binadj += flip # flip binadj to reflect closeness

            # print(f'Binadj dtype {binadj.dtype}')    

            # indices = binadj.nonzero().cpu().numpy()
            # i = list(zip(*indices))
            # vals = binadj[i].flatten().cpu().numpy()
            # binadj = torch.sparse_coo_tensor(i, vals, binadj.shape, device=device_list[-1], dtype=torch.half)

            binadj = binadj.cpu().numpy()
            binadj = np.where(binadj==0, None, binadj)
            np.fill_diagonal(binadj, 0)
            
            binadj = coo_matrix(binadj) # convert to sparse COO format matrix
            # binadj = tril(binadj.cpu(), format="coo") # convert to sparse COO format matrix
            
            # print(list(zip(*binadj.nonzero())))

            if args.verbose:
                print(f'\n The dimension of the COO distance matrix is {binadj.data.shape}\n')
                if binadj.data.shape[0] != 0:
                    print(f'Binadj mean {np.nanmean(binadj.data):.4f}, min {np.nanmin(binadj.data):.4f}, max {np.nanmax(binadj.data):.4f}')
                else:
                    print(f'Binadj empty! \n')
            
            comp_time = time.time()
            dgm = ripser_parallel(binadj, metric="precomputed", maxdim=UPPER_DIM, n_threads=-1, collapse_edges=True)
            comp_time = time.time() - comp_time

            torch.cuda.empty_cache()
            del binadj, flip
            
            total_time += comp_time
            print(f'\n Computation time: {comp_time/60:.5f} minutes \n')

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
        
        del activs, adj
        torch.cuda.empty_cache()

        betti_nums_list = np.array(betti_nums_list)
        betti_nums_list_3d = np.array(betti_nums_list_3d)
        
        # Plot betti numbers per dimension at current eps. thresh.
        make_plots(betti_nums_list, betti_nums_list_3d, epoch, num_nodes, thresholds, args.eps_thresh, CURVES_DIR, THREED_IMG_DIR, start, stop)

    print(f'\n Total computation time: {total_time/60:.5f} minutes \n')
