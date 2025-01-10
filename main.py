import argparse
import os
from utils import visible_print


parser = argparse.ArgumentParser()

parser.add_argument('--train', default=1, type=int)
parser.add_argument('--build_graph', default=1, type=int)
parser.add_argument('--post_process', default=1, type=int)
parser.add_argument('--net', help='Specify deep network architecture.')
parser.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)')
parser.add_argument('--subset', default=500, type=int, help='Subset size for building graph.')
parser.add_argument('--average', default=1, type=int, help='Average over all samples.')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: "spearman", "dcorr", or callable.')
parser.add_argument('--n_epochs_train', default='50', help='Number of epochs to train.')
parser.add_argument('--lr', default='0.001', help='Specify learning rate for training.')
parser.add_argument('--optimizer', default='adabelief', help='Define optimizer for training: "adabelief" or "adam"')
parser.add_argument('--epochs_test', default='0 4 8 20 30 40 50', help='Epochs for which you want to build graph.')
parser.add_argument('--thresholds', default='0.05 1.0', help='Define thresholds range in the form \'start stop\' ')
parser.add_argument('--reduction', default=None, type=str, help='Reductions: pca, umap or kmeans.')
parser.add_argument('--it', type=int, default=0)
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--resume_epoch', default=20, type=int)
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--save_dir', default='./results', help='Directory to save results.')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

ONAME = f'{args.net}_{args.dataset}_ss{args.it}' if args.dataset.__eq__('imagenet') else f'{args.net}_{args.dataset}'
SAVE_DIR = os.path.join(args.save_dir, ONAME)

if args.train:
    visible_print('Training network')

    cmd = f'python ./train.py'
    cmd += f' --net {args.net}'
    cmd += f' --dataset {args.dataset}'
    cmd == f' --epochs {args.n_epochs_train}'
    cmd += f' --lr {args.lr}'
    cmd += f' --it {args.it}'
    cmd += f' --chkpt_epochs {args.epochs_test}'
    cmd += f' --optimizer {args.optimizer}'
    cmd += f' --resume {args.resume}'
    cmd += f' --resume_epoch {args.resume_epoch}'
    cmd += f' --reduction {args.reduction}' if args.reduction else ''
    cmd += f' --metric {args.metric}' if args.metric else ''
    
    os.system(cmd)

if args.build_graph:
    visible_print('Building graph')
    
    cmd = f'python ./build_graph_functional.py'
    cmd += f' --net {args.net}'
    cmd += f' --dataset {args.dataset}'
    cmd += f' --chkpt_epochs {args.epochs_test}'
    cmd += f' --it {args.it}'
    cmd += f' --verbose {args.verbose}'
    cmd += f' --subset {args.subset}'
    cmd += f' --resume {args.resume}'
    cmd += f' --resume_epoch {args.resume_epoch}'
    cmd += f' --reduction {args.reduction}' if args.reduction else ''
    cmd += f' --metric {args.metric}' if args.metric else ''

    os.system(cmd)

if args.post_process:
    visible_print('Post-processing')

    cmd = f'python ./post_process.py'
    cmd += f' --net {args.net}'
    cmd += f' --dataset {args.dataset}'
    cmd += f' --save_dir {SAVE_DIR}'
    cmd += f' --chkpt_epochs {args.epochs_test}'
    cmd += f' --it {args.it}'
    cmd += f' --reduction {args.reduction}' if args.reduction else ''
    cmd += f' --metric {args.metric}' if args.metric else ''

    os.system(cmd)
