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
from numpy import inf
from config import SAVE_PATH
import os
import matplotlib.pyplot as plt
import gc

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
parser.add_argument('--save_epochs', nargs='+', type=int)

args = parser.parse_args()

SAVE_EPOCHS = list(args.save_epochs) # At what epochs to save train/test stats

ONAME = args.net + '_' + args.dataset + '_ss' + str(args.iter) # Meta-name to be used as prefix on all savings

summary_path = SAVE_PATH + '/' + args.net +  "/summary/"
if not os.path.isdir(summary_path):
        os.makedirs(summary_path)

summary_file = summary_path + ONAME + ".txt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

criterion  = get_criterion(args.dataset)

''' Build models '''
print('==> Building model..', "\n")
net = get_model(args.net, args.dataset)
# print(net)
net = net.to(device)
   
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

''' Initialize weights from checkpoint '''
if args.resume:
    net, best_acc, start_acc = init_from_checkpoint(net)
  
''' Optimization '''
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode='max', verbose=True)

''' Define passer '''
passer_train = Passer(net, train_loader, criterion, device)
passer_test = Passer(net, test_loader, criterion, device)

''' Define manipulator '''
manipulator = load_manipulator(args.permute_labels)

''' Make intial pass before any training '''
loss_te, acc_te = passer_test.run()

save_checkpoint(checkpoint = {'net':net.state_dict(), 'acc': acc_te, 'epoch': 0}, path='./checkpoint/' + args.net + '/' + ONAME + '/', fname='ckpt_trial_' + str(args.trial) + '_epoch_0.t7')

print("Begin training", "\n")

losses = []
best_acc_tr = -inf
best_acc_te = -inf
best_epoch_tr = 0
best_epoch_te = 0
for epoch in range(start_epoch, start_epoch + args.epochs):
    print('Epoch {}'.format(epoch))

    loss_tr, acc_tr = passer_train.run(optimizer, manipulator=manipulator)
    loss_te, acc_te = passer_test.run()

    if acc_tr > best_acc_tr:
        best_acc_tr = acc_tr
        best_epoch_tr = epoch

    if acc_te > best_acc_te:
        best_acc_te = acc_te
        best_epoch_te = epoch

    with open(summary_file, 'w') as f:
        lines = ["Train epoch: " + str(best_epoch_tr) + "\n", "Train acc: " + str(best_acc_tr) + "\n",
                 "Test epoch: " + str(best_epoch_te) + "\n", "Test acc: " + str(best_acc_te) + "\n"]
        f.writelines(lines)

    losses.append({'loss_tr': loss_tr, 'loss_te': loss_te, 'acc_tr': acc_tr, 'acc_te': acc_te, 'epoch': int(epoch)})
    lr_scheduler.step(acc_te)

    if epoch in SAVE_EPOCHS:
        save_checkpoint(checkpoint = {'net':net.state_dict(), 'acc': acc_te, 'epoch': epoch}, path='./checkpoint/' + args.net + '/' + ONAME + '/', fname='ckpt_trial_' + str(args.trial) + '_epoch_' + str(epoch) + '.t7')

    gc.collect()

'''Save losses'''
save_losses(losses, path='./losses/' + args.net + '/' + ONAME + '/', fname='stats_trial_' + str(args.trial) + '.pkl')

'''Make plots'''
X = [loss['epoch'] for loss in losses]
directory = os.path.join(summary_path, ONAME)

'''Create plots of accuracies'''
plt.xlabel('Epoch (N)')
plt.ylabel('Accuracy')
test_acc = [loss['acc_te']/100. for loss in losses]
train_acc = [loss['acc_tr']/100. for loss in losses]
plt.plot(X, test_acc, label='Test')
plt.plot(X, train_acc, label='Train')
plt.legend()
plt.title('Accuracy v. Epoch')
plt.savefig(directory + "_acc.png")
plt.clf()

'''Create plots of losses'''
plt.xlabel('Epoch (N)')
plt.ylabel('Loss')
test_loss = np.array([np.mean(loss['loss_te']) for loss in losses])
test_std = np.array([np.std(loss['loss_te']) for loss in losses])
argmin_test_loss = np.argmin(test_loss)
min_test_loss = test_loss[argmin_test_loss]
plt.fill_between(X, test_loss - test_std, test_loss + test_std, alpha=0.1, interpolate=True)

train_loss = np.array([np.mean(loss['loss_tr']) for loss in losses])
train_std = np.array([np.std(loss['loss_tr']) for loss in losses])
plt.fill_between(X, train_loss - train_std, train_loss + train_std, alpha=0.1, interpolate=True)
plt.vlines(X[argmin_test_loss], 0, min_test_loss, linestyles='dashed', label='Min Test Loss')

plt.plot(X, test_loss, label='Test Mean')
plt.plot(X, train_loss, label='Train Mean')
plt.legend()
plt.title('Average Loss v. Epoch')
plt.savefig(directory + "_loss.png")
plt.clf()

gc.collect()
