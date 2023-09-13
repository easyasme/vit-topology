import argparse
import os

from config import SAVE_PATH

parser = argparse.ArgumentParser()

parser.add_argument('--train', default=1, type=int)
parser.add_argument('--build_graph', default=1, type=int)
parser.add_argument('--comp_topo', default=1, type=int)
parser.add_argument('--net', help='Specify deep network architecture (e.g. lenet, alexnet, resnet, inception, vgg, etc)')
parser.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)')
parser.add_argument('--trial', default=0, help='Specify trial number. Used to differentiate btw multiple trainings of same setup.')
parser.add_argument('--n_epochs_train', default='10', help='Number of epochs to train.')
parser.add_argument('--lr', default='0.01', help='Specify learning rate for training.')
parser.add_argument('--permute_labels', default='0.0', help='Specify if labels are going to be permuted. Float between 0 and 1. If 0, no permutation. If 1 all labels are permuted. Otherwise proportion of labels.')
parser.add_argument('--epochs_test', help='Epochs for which you want to build graph. String of positive natural numbers separated by spaces.')
parser.add_argument('--thresholds', default='0.5 1.0', help='Defining thresholds range in the form \'start step stop \' ')
parser.add_argument('--filtration', default='nominal')
parser.add_argument('--scale', default='linear')
parser.add_argument('--split',  default='0', help='Split network into partitions.')
parser.add_argument('--subsplit',  default='0', help='Split each partition.')
parser.add_argument('--kl', default='0', help='TO ADD.')
parser.add_argument('--graph_type', default='functional')
parser.add_argument('--n_samples', type=int, default=5)
parser.add_argument('--select_nodes', default='0')
parser.add_argument('--partition', default='hardcoded')
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
    os.system('python ./train.py --net '+args.net+' --dataset '+args.dataset+' --trial '+args.trial+' --epochs '+args.n_epochs_train+' --lr '+args.lr+' --permute_labels '+args.permute_labels+' --iter '+str(args.iter)+' --save_epochs '+args.epochs_test)

if args.build_graph:
    visible_print('Building '+args.graph_type+' graph')
    os.system('python ./build_graph_functional.py --save_path '+SAVE_PATH+'/'+args.net+'/'+' --net '+args.net+' --dataset '+args.dataset+' --trial '+args.trial+' --epochs '+args.epochs_test+' --filtration '+args.filtration+' --split '+args.split+' --kl '+args.kl+' --permute_labels '+args.permute_labels+' --iter '+str(args.iter))

if args.comp_topo:
    visible_print('Computing topology')
    os.system('python ./compute_topology.py --save_path '+SAVE_PATH+'/'+args.net+'/'+' --net '+args.net+' --dataset '+args.dataset+' --epochs '+args.epochs_test+' --trial '+args.trial+' --iter '+str(args.iter))
