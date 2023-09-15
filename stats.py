import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os

PKL_FILE = '/stats_trial_0.pkl'
NET = 'lenetext'

path = 'losses/' + NET + '/'
save_path = './results/' + NET + '/acc_stats/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load data
all_acc_te = []
all_acc_tr = []
# all_loss_te = []
# all_loss_tr = []
for fl in os.listdir(path):
    pk_f = os.path.join(path, fl + PKL_FILE)
    print(fl)
    with open(pk_f, 'rb') as f:
        losses = pkl.load(f)

        acc_te = np.array([loss['acc_te'] for loss in losses]) # test accuracy per epoch
        acc_tr = np.array([loss['acc_tr'] for loss in losses]) # train accuracy per epoch

        # loss_te = np.array([loss['loss_te'] for loss in losses]) # test loss per epoch
        # loss_tr = np.array([loss['loss_tr'] for loss in losses]) # train loss per epoch

        all_acc_te.append(acc_te)
        all_acc_tr.append(acc_tr)
        # all_loss_te.append(loss_te)
        # all_loss_tr.append(loss_tr)

# Compute stats
all_acc_te = np.array(all_acc_te)
all_acc_tr = np.array(all_acc_tr)
# all_loss_te = np.array(all_loss_te)
all_loss_tr = np.array(all_loss_tr)

mean_acc_te = np.mean(all_acc_te, axis=0)
mean_acc_tr = np.mean(all_acc_tr, axis=0)
# mean_loss_te = np.mean(all_loss_te, axis=(0, 2))
# mean_loss_tr = np.mean(all_loss_tr, axis=(0, 2))


std_acc_te = np.std(all_acc_te, axis=0)
std_acc_tr = np.std(all_acc_tr, axis=0)
# std_loss_te = np.std(all_loss_te, axis=(0, 2))
# std_loss_tr = np.std(all_loss_tr, axis=(0, 2))

# Plot
X = np.arange(0, len(mean_acc_te))

# Loss
plt.xlabel('Epoch (N)')
plt.ylabel('Average Accuracy')
plt.fill_between(X, mean_acc_te - std_acc_te, mean_acc_te + std_acc_te, alpha=0.1)
plt.fill_between(X, mean_acc_tr - std_acc_tr, mean_acc_tr + std_acc_tr, alpha=0.1)
plt.plot(X, mean_acc_te, label='Avg. Test')
plt.plot(X, mean_acc_tr, label='Avg. Train')
plt.legend()
plt.title('Average Accuracy over Datasets')
plt.savefig(save_path + "avg_acc.png")
plt.clf()

# Accuracy
# plt.xlabel('Epoch (N)')
# plt.ylabel('Average Loss')
# plt.fill_between(X, mean_loss_te - std_loss_te, mean_loss_te + std_loss_te, alpha=0.1)
# plt.fill_between(X, mean_loss_tr - std_loss_tr, mean_loss_tr + std_loss_tr, alpha=0.1)
# plt.plot(X, mean_loss_te, label='Avg. Test')
# plt.plot(X, mean_loss_tr, label='Avg. Train')
# plt.legend()
# plt.title('Average Loss over Datasets')
# plt.savefig(save_path + "avg_loss.png")
# plt.clf()