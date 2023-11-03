from __future__ import print_function
import torch.backends.cudnn as cudnn
from utils import *
from models.utils import get_model
from passers import Passer
from loaders import *
from graph import *
import os

parser = argparse.ArgumentParser()

parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--save_path', default='./results', type=str)
parser.add_argument('--epochs', nargs='+', action='extend', type=int, default=[1, 5, 10, 20])
parser.add_argument('--input_size', default=32, type=list)
parser.add_argument('--thresholds', default='0.5 1.0', help='Defining thresholds range in the form \'start stop\' ')
parser.add_argument('--iter', type=int, default=0)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
'''Meta-name to be used as prefix on all savings'''
SAVE_DIR = os.path.join(args.save_path, args.net + '_' + args.dataset + '_' + 'ss' + str(args.iter) + '/bin/')

'''Define layer to start feature extraction'''
START_LAYER = 3 if args.net in ['vgg', 'resnet'] else 0 

''' If save directory doesn't exist create '''
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)    

# Build models
print('==> Building model..', "\n")
net = get_model(args.net, args.dataset)
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
''' Prepare test data loader '''
functloader = loader(args.dataset+'_test', batch_size=100, iter=args.iter, verbose=False)

''' Prepare criterion '''
criterion = nn.CrossEntropyLoss()

# Create a list of thresholds from start to stop with only 
# 10 elements (num=10)
start = float(args.thresholds.split(' ')[0])
stop = float(args.thresholds.split(' ')[1])

for epoch in args.epochs:
    print("\n", '==> Loading checkpoint for epoch {}...'.format(epoch), "\n")

    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        
    checkpoint = torch.load('./checkpoint/'+ args.net + '/' + args.net + '_' + args.dataset + '_' + 'ss' + str(args.iter) + '/ckpt_epoch_{}.t7'.format(epoch))

    # have to rename keys because of DataParallel wrapper;
    # this is also why we need to add the 'module' prefix when
    # extracting forward_features in passers.py
    checkpoint['net'] = {f'module.{k}': v for k, v in checkpoint['net'].items()}
        
    net.load_state_dict(checkpoint['net'])
        
    ''' Define passer and get activations '''
    passer = Passer(net, functloader, criterion, device)
    activs = passer.get_function()
    activs = signal_concat(activs)
    adj = adjacency(activs)
    
    print('The dimension of the adjacency matrix is {}'.format(adj.shape))
    print('Adj mean {}, min {}, max {}'.format(np.mean(adj), np.min(adj), np.max(adj)))

    for t in np.linspace(start=start, stop=stop, num=10):
        # ''' Write adjacency to binary. To use as DIPHA input for persistence homology '''
        save_dipha(SAVE_DIR + 'adj_epc{}_thresh{:.2f}.bin'.format(epoch, t), adj > t)