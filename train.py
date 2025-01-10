from __future__ import print_function

import argparse
import gc
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from adabelief_pytorch import AdaBelief

from loaders import *
from models.utils import get_model, init_from_checkpoint
from passers import Passer
from savers import save_checkpoint, save_losses
from PIL import Image

from config import SEED, IMG_SIZE

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--reduction', default=None, type=str, help='Reductions: "pca" or "umap"')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: "spearman", "dcorr", or callable.')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--optimizer', default='adabelief', type=str, help='optimizer')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume from epoch')
parser.add_argument('--train_batch_size', default=32, type=int)
parser.add_argument('--val_batch_size', default=32, type=int)
parser.add_argument('--it', default=0, type=int)
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])

args = parser.parse_args()

ONAME = f'{args.net}_{args.dataset}_ss{args.it}' if args.dataset == 'imagenet' else f'{args.net}_{args.dataset}'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Training device: ", device, "\n")

''' Build models '''
print(f'==> Building model..\n')
net = get_model(args.net, args.dataset)
net = net.to(device)

transform = net._get_transform()
print(f'Transform: {transform}')

''' Prepare loaders '''
print(f'==> Preparing data..\n')
print(f'Preparing train loader')
train_loader = loader(f'{args.dataset}_train', batch_size=args.train_batch_size, it=args.it, verbose=True, transform=transform)

print(f'Preparing val loader\n')
val_loader = loader(f'{args.dataset}_val', batch_size=args.val_batch_size, it=args.it, verbose=False, transform=transform)

''' Define loss function '''
criterion = nn.CrossEntropyLoss()

''' Optimization '''
if args.optimizer == 'adabelief':
    optimizer = AdaBelief(net.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-2, weight_decouple=True, rectify=False, fixed_decay=False, amsgrad=False)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

best_acc = 0  # best val accuracy
start_epoch = 1  # start from epoch 1 or last checkpoint epoch

''' Initialize from checkpoint '''
if args.resume:
    net, optimizer, loss_tr, loss_te, acc_tr, acc_te, start_epoch = init_from_checkpoint(net, optimizer, args)
    start_epoch += 1
    print(f'==> Resuming from checkpoint.. Epoch: {start_epoch}\n')
elif args.it != 0:
    net, optimizer, loss_tr, loss_te, acc_tr, acc_te, start_epoch = init_from_checkpoint(net, optimizer, args, start=True)
    start_epoch += 1

''' Define passer '''
passer_train = Passer(net, train_loader, criterion, device)
passer_val = Passer(net, val_loader, criterion, device)

''' Make intial pass before any training '''
if not args.resume and args.it == 0:
    loss_te, acc_te = passer_val.run()
    save_checkpoint(checkpoint = {'net':net.state_dict(),
                                'loss_te': loss_te,
                                'acc_te': acc_te,
                                'optimizer': optimizer.state_dict()},
                                path=f'./checkpoint/{args.net}/{ONAME}/', fname=f"ckpt_epoch_0.pt")


losses = []
for epoch in range(start_epoch, args.epochs + 1):
    print(f'Epoch {epoch}')

    loss_tr, acc_tr = passer_train.run(optimizer)
    loss_te, acc_te = passer_test.run()

    losses.append({'loss_tr': loss_tr,
                   'loss_te': loss_te,
                   'acc_tr': acc_tr,
                   'acc_te': acc_te,
                   'epoch': int(epoch)})

    if epoch in vars(args)['chkpt_epochs']:
        save_checkpoint(checkpoint = {'net':net.state_dict(),
                                      'loss_tr': loss_tr,
                                      'loss_te': loss_te,
                                      'acc_tr': acc_tr,
                                      'acc_te': acc_te,
                                      'epoch': epoch,
                                      'optimizer': optimizer.state_dict()},
                                       path=f'./checkpoint/{args.net}/{ONAME}/', fname=f"ckpt_epoch_{epoch}.pt")

    gc.collect()

'''Save losses'''
path = f'./losses/{args.net}/{ONAME}'
path += f'/{args.reduction}' if args.reduction else ''
path += f'/{args.metric}' if args.metric else ''
save_losses(losses, path=path, fname='/stats.pkl')
