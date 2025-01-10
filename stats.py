import os
import pickle
import plotly.graph_objects as go
from plotly.io import write_image

from PIL import Image

from loaders import *

NET = 'resnet'
DATASET = 'imagenet'
START = 0
SUBSETS = 30 # 1 for none, 30 for all

SAVE_DIR = './results/stats'
RED = 'kmeans' # 'pca' or 'umap' or None
METRIC = 'spearman' # 'euclidean' or 'cosine' or None

SAVE_DIR += f'/{RED}' if RED is not None else ''
SAVE_DIR += f'/{METRIC}' if METRIC is not None else ''

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    
    return dst

def get_concat_v_multi_blank(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_v(_im, im)
    
    return _im

time_only = False

# ''' Make plots of losses and accuracies '''
Xs = None
test_accs = []
train_accs = []
times = []
for it in range(START, SUBSETS):
    # print(f'Processing losses for subset {it}')
    if DATASET.__eq__('imagenet'):
        # pkl_path = f"./losses/{NET}/{NET}_{DATASET}_ss{it}/"
        pkl_path = f'/home/trogdent/compute/qual/results/trial_0/losses/{NET}/{NET}_{DATASET}_ss{it}/'
    else:
        # pkl_path = f"./losses/{NET}/{NET}_{DATASET}/"
        pkl_path = f'/home/trogdent/compute/qual/results/trial_0/losses/{NET}/{NET}_{DATASET}/'

    pkl_path += f'{RED}/' if RED is not None else ''
    pkl_path += f'{METRIC}/' if METRIC is not None else ''
    time_pkl_path = pkl_path + 'time.pkl'
    pkl_path += 'stats.pkl'

    if not os.path.exists(pkl_path):
        print("No stats.pkl found at", pkl_path)
        continue
    
    if DATASET.__eq__('imagenet'):
        save_dir = f"{SAVE_DIR}/{NET}_{DATASET}_ss{it}/images/loss/"
    else:
        save_dir = f'{SAVE_DIR}/{NET}_{DATASET}/images/loss/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    acc_save_file = f"{save_dir}/acc.png"
    loss_save_file = f"{save_dir}/loss.png"

    with open(time_pkl_path, 'rb') as f:
        time = pickle.load(f)
    times.append(time)
    
    if not time_only:
        with open(pkl_path, 'rb') as f:
            losses = pickle.load(f)
            
        X = [loss['epoch'] for loss in losses]
        Xs = X

        test_accs.append([loss['acc_te']/100. for loss in losses])
        train_accs.append([loss['acc_tr']/100. for loss in losses])

        '''Create plots of accuracies'''
        fig = go.Figure(layout=go.Layout(
                        title=f'Accuracy on subset {it}',
                        xaxis=dict(title='Epoch'),
                        yaxis=dict(title='Accuracy'),
                        font=dict(size=16)
                        )
                    )
        fig.add_trace(go.Scatter(x=X, y=[loss['acc_te']/100. for loss in losses], mode='lines', line_color='red', name='Test'))
        fig.add_trace(go.Scatter(x=X, y=[loss['acc_tr']/100. for loss in losses], mode='lines', line_color='blue', name='Train'))

        write_image(fig, acc_save_file, format='png')

        '''Create plots of losses'''
        test_loss = np.array([np.mean(loss['loss_te']) for loss in losses])
        test_std = np.array([np.std(loss['loss_te']) for loss in losses])
        train_loss = np.array([np.mean(loss['loss_tr']) for loss in losses])
        train_std = np.array([np.std(loss['loss_tr']) for loss in losses])

        fig = go.Figure(layout=go.Layout(
                        title=f'Average loss on subset {it}',
                        xaxis=dict(title='Epoch'),
                        yaxis=dict(title='Loss'),
                        font=dict(size=16)
                        ))
        fig.add_trace(go.Scatter(x=X, y=test_loss, mode='lines', fill=None, line_color='red', name='Test'))
        fig.add_trace(go.Scatter(x=X, y=test_loss + test_std, fill=None, mode='lines', showlegend=False, line=dict(color='red', width=.1, dash='dash')))
        fig.add_trace(go.Scatter(x=X, y=test_loss - test_std, fill='tonexty', mode='lines', showlegend=False, line=dict(color='red', width=.1, dash='dash')))
        
        fig.add_trace(go.Scatter(x=X, y=train_loss, mode='lines', fill=None, line_color='blue', name='Train'))
        fig.add_trace(go.Scatter(x=X, y=train_loss + train_std, fill=None, mode='lines', showlegend=False, line=dict(color='blue', width=.1, dash='dash')))
        fig.add_trace(go.Scatter(x=X, y=train_loss - train_std, fill='tonexty', mode='lines', showlegend=False, line=dict(color='blue', width=.1, dash='dash')))
        
        # fig.update_layout(title=f'Average loss on subset {it}', xaxis_title='Epoch', yaxis_title='Loss')
        write_image(fig, loss_save_file, format='png')

''' Plot averages over all subsets '''
if not time_only:
    save_dir = f"{SAVE_DIR}/{NET}_{DATASET}/images/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    avg_acc_save_file = f"{SAVE_DIR}/{NET}_{DATASET}/images/avg_acc.png"

    test_accs = np.array(test_accs)
    test_mean = np.mean(test_accs, axis=0)
    test_std = np.std(test_accs, axis=0)

    train_accs = np.array(train_accs)
    train_mean = np.mean(train_accs, axis=0)
    train_std = np.std(train_accs, axis=0)

    fig = go.Figure(layout=go.Layout(
                    title=None,
                    margin=dict(t=7.5),
                    xaxis=dict(title='Epoch'),
                    yaxis=dict(title='Accuracy'),
                    font=dict(
                        size=16)
                    )
                )
    fig.add_trace(go.Scatter(x=Xs, y=test_mean, mode='lines', line_color='red', name='Test'))
    fig.add_trace(go.Scatter(x=Xs, y=test_mean + test_std, fill=None, mode='lines', showlegend=False, line=dict(color='red', width=.1, dash='dash')))
    fig.add_trace(go.Scatter(x=Xs, y=test_mean - test_std, fill='tonexty', mode='lines', showlegend=False, line=dict(color='red', width=.1, dash='dash')))

    fig.add_trace(go.Scatter(x=Xs, y=train_mean, mode='lines', line_color='blue', name='Train'))
    fig.add_trace(go.Scatter(x=Xs, y=train_mean + train_std, fill=None, mode='lines', showlegend=False, line=dict(color='blue', width=.1, dash='dash')))
    fig.add_trace(go.Scatter(x=Xs, y=train_mean - train_std, fill='tonexty', mode='lines', showlegend=False, line=dict(color='blue', width=.1, dash='dash')))

    # fig.update_layout(title=f'Average accuracy over subsets {START}-{SUBSETS-1}', xaxis_title='Epoch', yaxis_title='Accuracy', yaxis=dict(tickfont=dict(size=20)), xaxis=dict(tickfont=dict(size=20)))
    write_image(fig, avg_acc_save_file, format='png')

times = np.array(times)
print(f'Average time per subset: {times.mean()/60.:.3f} minutes')

# ''' Concatenate images of curves '''
# for it in range(START, SUBSETS):
#     print(f'Processing images for subset {it}')
#     if DATASET.__eq__('imagenet'):
#         save_dir = f"{SAVE_DIR}/{NET}_{DATASET}_ss{it}/images/curves/"
#     else:
#         save_dir = f'{SAVE_DIR}/{NET}_{DATASET}/images/curves/'
    
#     if not os.path.exists(save_dir):
#         print("No directory", save_dir)
#         continue

#     files = os.listdir(save_dir)
#     files = [f for f in files if f.startswith('epoch')]

#     if len(files) == 0:
#         print("No images found at", save_dir)
#         continue
#     files.sort(key=lambda x: int(x.split('_')[1])) # sort by epoch
    
#     images = [Image.open(f"{save_dir}/{f}") for f in files] # open images
#     get_concat_v_multi_blank(images).save(f"{save_dir}/concat.png") # concat and save
    