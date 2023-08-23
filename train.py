from __future__ import print_function
import argparse
from passers import Passer
from savers import save_checkpoint, save_losses
from loaders import *
from labels import *
from models.utils import get_model, get_criterion, init_from_checkpoint
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
parser.add_argument('--resume_epoch', default=20, type=int, help='resume from epoch')
parser.add_argument('--save_every', default=1, type=int)
parser.add_argument('--permute_labels', default=0, type=float)
parser.add_argument('--fixed_init', default=0, type=int)
parser.add_argument('--train_batch_size', default=128, type=int)
parser.add_argument('--test_batch_size', default=100, type=int)
parser.add_argument('--input_size', default=32, type=int)
parser.add_argument('--iter', default=0, type=int)

args = parser.parse_args()

SAVE_EPOCHS = list(range(11)) + list(range(10, args.epochs + 1, args.save_every)) # At what epochs to save train/test stats
ONAME = args.net + '_' + args.dataset # Meta-name to be used as prefix on all savings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device, "\n")

best_acc = 0  # best test accuracy
start_epoch = 1  # start from epoch 1 or last checkpoint epoch

''' Prepare loaders '''
print('==> Preparing data..', "\n")
print("Preparing train loader", "\n")
train_loader = loader(args.dataset + '_train', batch_size=args.train_batch_size, iter=args.iter)
print("Preparing test loader")
test_loader = loader(args.dataset + '_test', batch_size=args.test_batch_size, iter=args.iter)

n_samples = len(train_loader) * args.train_batch_size

criterion  = get_criterion(args.dataset)

''' Build models '''
print('==> Building model..')
net = get_model(args.net, args.dataset)
print(net)
net = net.to(device)
   
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

''' Initialize weights from checkpoint '''
if args.resume:
    net, best_acc, start_acc = init_from_checkpoint(net)
  
''' Optimization '''
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4, amsgrad=True)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode='max', verbose=True)

''' Define passer '''
passer_train = Passer(net, train_loader, criterion, device)
passer_test = Passer(net, test_loader, criterion, device)

''' Define manipulator '''
manipulator = load_manipulator(args.permute_labels)

''' Make intial pass before any training '''
loss_te, acc_te = passer_test.run()

save_checkpoint(checkpoint = {'net':net.state_dict(), 'acc': acc_te, 'epoch': 0}, path='./checkpoint/' + ONAME + '/', fname='ckpt_trial_' + str(args.trial) + '_epoch_0.t7')

losses = []
for epoch in range(start_epoch, start_epoch + args.epochs):
    print('Epoch {}'.format(epoch))

    loss_tr, acc_tr = passer_train.run(optimizer, manipulator=manipulator)
    loss_te, acc_te = passer_test.run()
   
    losses.append({'loss_tr':loss_tr, 'loss_te': loss_te, 'acc_tr': acc_tr, 'acc_te':acc_te})
    lr_scheduler.step(acc_te)

    if epoch in SAVE_EPOCHS:
        save_checkpoint(checkpoint = {'net':net.state_dict(), 'acc': acc_te, 'epoch': epoch}, path='./checkpoint/' + ONAME + '/', fname='ckpt_trial_' + str(args.trial) + '_epoch_' + str(epoch) + '.t7')

'''Save losses'''
save_losses(losses, path='./losses/' + ONAME + '/', fname='stats_trial_' + str(args.trial) + '.pkl')
