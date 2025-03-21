import os
import os.path
import pickle
import sys
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from matplotlib import cm

from config import UPPER_DIM


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except ValueError:
    term_width = 0
    
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def create_img_from_tensors(features):
    ''' Create PNG images from tensors'''
    for idx, activation in enumerate(torch.unbind(features, dim=0)):
        print(f"Encoder Block {idx} activation shape: {activation.shape}")
        plt.imshow(activation.permute(1, 2, 0).numpy(force=True))
        plt.savefig(f"encoder_block_{idx}_activation.png")

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_accuracy(predictions, targets):
    ''' Compute accuracy of predictions to targets. max(predictions) is best'''
    _, predicted = predictions.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()

    return 100. * (correct / total)

def get_device_list():
    device_list = []
    if torch.cuda.device_count() > 1:
        device_list = [torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count())]
        print("Using", torch.cuda.device_count(), "GPUs")
        for i, device in enumerate(device_list):
            print(f"Device {i}: {device}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_list.append(device)
        print(f'Using {device}')

    return device_list

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def make_plots(betti_nums, betti_nums_3d, epoch, num_nodes, orig_nodes, thresholds, eps_thresh, curves_dir, threeD_img_dir, start, stop, net, dataset, subset):
    # pickle betti numbers along with epoch and thresholds in a dictionary
    betti_nums_dict = {'epoch': epoch, 'thresholds': thresholds, 'betti_nums': betti_nums, 'num_nodes': num_nodes, 'orig_nodes': orig_nodes}
    
    if dataset == 'mnist':
        pkl_file = f'./losses/{net}/{net}_{dataset}/betti_nums.pkl'
    else:
        pkl_file = f'./losses/{net}/{net}_{dataset}_ss{subset}/betti_nums.pkl'

    os.makedirs(os.path.dirname(pkl_file), exist_ok=True)

    with open(pkl_file, 'wb') as f:
        pickle.dump(betti_nums_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    color = {0: 'mediumseagreen', 1: 'b', 2: 'r', 3: 'g', 4: 'c', 5: 'k', 6: 'm', 7: 'w'}
    for i in range(0, UPPER_DIM+1):
        bn_img_path = curves_dir + f'/epoch_{epoch}_dim_{UPPER_DIM}_bn_{i}.png'
            
        fig = plt.figure()

        plt.plot(thresholds, betti_nums[:, i] / num_nodes, label=f'Betti {i}', color=color[i])

        max_idx = np.argmax(betti_nums[:, i] / num_nodes)
        max_val = betti_nums[max_idx, i] / num_nodes
        plt.vlines(x=thresholds[max_idx], ymin=0, ymax=max_val, color='orange', linestyle='dashed', label=f'Max loc. {thresholds[max_idx]:.3f}')
        plt.hlines(y=max_val, xmin=start, xmax=stop, color='orange', linestyle='dashed', label=f'Max val. {max_val:.3f}')

        plt.xlabel('Thresholds')
        plt.ylabel('Betti Numbers')
        plt.ylim(0, 1.05* max_val)
        plt.grid()
        plt.title(f'Epoch {epoch}')
        plt.legend()
            
        fig.savefig(bn_img_path)
            
        plt.clf()
        plt.close(fig)

    for i in range(0, UPPER_DIM+1):
        bn3d_img_path = threeD_img_dir + f'/epoch_{epoch}_dim_{UPPER_DIM}_bn_{i}_3d.pkl'

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        X, Y = np.meshgrid(eps_thresh, thresholds)
        Z = betti_nums_3d[:,:,i] / num_nodes
        ax.plot_surface(X, Y, Z, cmap=cm.Spectral, alpha=0.8 , label=f'Betti {i}')

        ax.set_xlabel('Eps Thresholds')
        ax.set_ylabel('Thresholds')
        ax.set_zlabel('Betti Numbers')
        ax.set_title(f'Epoch {epoch}')

        with open(bn3d_img_path, 'wb') as f:
            pickle.dump(fig, f, protocol=pickle.HIGHEST_PROTOCOL)

        plt.clf()
        plt.close(fig)

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * (current/total))
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def visible_print(message):
    ''' Visible print'''
    print('')
    print(50*'-')
    print(message)
    print(50*'-')
    print('')