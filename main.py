import argparse
import os

from config import SAVE_PATH

parser = argparse.ArgumentParser()

parser.add_argument('--train', default=1, type=int)
parser.add_argument('--build_graph', default=1, type=int)
parser.add_argument('--comp_topo', default=1, type=int)
parser.add_argument('--net', help='Specify deep network architecture.')
parser.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)')
parser.add_argument('--n_epochs_train', default='10', help='Number of epochs to train.')
parser.add_argument('--lr', default='0.01', help='Specify learning rate for training.')
parser.add_argument('--epochs_test', default='1 5 10', help='Epochs for which you want to build graph.')
parser.add_argument('--thresholds', default='0.5 1.0', help='Defining thresholds range in the form \'start stop\' ')
parser.add_argument('--filtration', default='nominal')
parser.add_argument('--graph_type', default='functional')
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()

def visible_print(message):
    ''' Visible print'''
    print('')
    print(50*'-')
    print(message)
    print(50*'-')
    print('')

if args.train:
    visible_print('Training network')
    os.system('python ./train.py --net '+args.net+' --dataset '+args.dataset+' --epochs '+args.n_epochs_train+' --lr '+args.lr+' --iter '+str(args.iter)+' --chkpt_epochs '+args.epochs_test)

if args.build_graph:
    visible_print('Building '+args.graph_type+' graph')
    os.system('python ./build_graph_functional.py --save_path '+SAVE_PATH+'/'+args.net+'/'+' --net '+args.net+' --dataset '+args.dataset+' --epochs '+args.epochs_test+' --filtration '+args.filtration+' --iter '+str(args.iter))

if args.comp_topo:
    visible_print('Computing topology')
    os.system('python ./compute_topology.py --save_path '+SAVE_PATH+'/'+args.net+'/'+' --net '+args.net+' --dataset '+args.dataset+' --epochs '+args.epochs_test+' --iter '+str(args.iter))
