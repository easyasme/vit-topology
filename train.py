from __future__ import print_function

import argparse
import gc

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loaders import *
from models.utils import get_model, init_from_checkpoint
from passers import Passer
from savers import save_checkpoint, save_losses

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--optimizer', default='adabelief', type=str, help='optimizer')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
parser.add_argument('--resume_epoch', default=20, type=int, help='resume from epoch')
parser.add_argument('--train_batch_size', default=32, type=int)
parser.add_argument('--test_batch_size', default=32, type=int)
parser.add_argument('--input_size', default=32, type=int)
parser.add_argument('--iter', default=0, type=int)
parser.add_argument('--chkpt_epochs', nargs='+', action='extend', type=int, default=[])


args = parser.parse_args()

if args.dataset == 'imagenet':
    ONAME = f'{args.net}_{args.dataset}_ss{args.iter}' # Meta-name to be used as prefix on all savings
else:
    ONAME = f'{args.net}_{args.dataset}'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device, "\n")

best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 1 or last checkpoint epoch

''' Prepare loaders '''
print('==> Preparing data..', "\n")
print("Preparing train loader")
train_loader = loader(args.dataset + '_train', batch_size=args.train_batch_size, iter=args.iter, verbose=True)
print("Preparing test loader", "\n")
test_loader = loader(args.dataset + '_test', batch_size=args.test_batch_size, iter=args.iter, verbose=True)

n_samples = len(train_loader) * args.train_batch_size

criterion = nn.CrossEntropyLoss()

''' Build models '''
print('==> Building model..', "\n")
net = get_model(args.net, args.dataset)
net = net.to(device)
   
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

''' Initialize weights from checkpoint '''
if args.resume:
    net, best_acc, start_acc = init_from_checkpoint(net)
  
''' Optimization '''
if args.optimizer == 'adabelief':
    optimizer = AdaBelief(net.parameters(), lr=args.lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-2, weight_decouple=True, rectify=False, fixed_decay=False, amsgrad=False)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

''' Learning rate scheduler '''
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode='max', verbose=True)

''' Define passer '''
passer_train = Passer(net, train_loader, criterion, device)
passer_test = Passer(net, test_loader, criterion, device)

''' Make intial pass before any training '''
loss_te, acc_te = passer_test.run()

save_checkpoint(checkpoint = {'net':net.state_dict(),
                              'acc': acc_te, 'epoch': 0},
                               path=f'./checkpoint/{args.net}/{ONAME}/', fname='ckpt_epoch_0.pt')

print("Begin training", "\n")

losses = []
for epoch in range(start_epoch, start_epoch + args.epochs):
    print('Epoch {}'.format(epoch))

    loss_tr, acc_tr = passer_train.run(optimizer)
    loss_te, acc_te = passer_test.run()

    losses.append({'loss_tr': loss_tr, 'loss_te': loss_te, 'acc_tr': acc_tr, 'acc_te': acc_te, 'epoch': int(epoch)})
    lr_scheduler.step(acc_te)

    if epoch in vars(args)['chkpt_epochs']:
        save_checkpoint(checkpoint = {'net':net.state_dict(), 'acc': acc_te, 'epoch': epoch}, path=f'./checkpoint/{args.net}/{ONAME}/', fname=f"ckpt_epoch_{epoch}.pt")

    gc.collect()

'''Save losses'''
save_losses(losses, path=f'./losses/{args.net}/{ONAME}/', fname='stats.pkl')
