import os
from .lenet import *
from .vgg import *
from .resnet import *
from .alexnet import *
from .densenet import *
from .inception import *
from .conv_x import * 
from .fcnet import *

from config import IMG_SIZE


def get_model(name, dataset):
    if name == 'conv_2':
        net = Conv_2(num_classes=10)
    elif name == 'conv_4':
        net = Conv_4(num_classes=10)
    elif name == 'conv_6':
        net = Conv_6(num_classes=10)

    elif name == 'fcnet':
        net = FCNet(input_size=3, num_classes=3)
    
    elif name=='lenet' and dataset == 'mnist':
        print("\n Fetching LeNet")
        print("Input size: ", 32, '\n')
        net = LeNet(num_channels=1, num_classes=10)
    elif name=='lenet' and dataset == 'imagenet':
        print("\n Fetching LeNet")
        print("Input size:", IMG_SIZE)
        net = LeNet(num_channels=3, num_classes=10, input_size=IMG_SIZE)
    
    elif name=='lenetext' and dataset=='imagenet':
        print("\n Fetching LeNetExt")
        print("Input size:", IMG_SIZE)
        net = LeNetExt(n_channels=3, num_classes=10, input_size=IMG_SIZE)
    elif name=='lenetext' and dataset=='mnist':
        print("\n Fetching LeNetExt")
        print("Input size: ", 32, '\n')
        net = LeNetExt(n_channels=1, num_classes=10)

    elif name=='vgg' and dataset=='imagenet':
        net = VGG('VGG16', num_classes=10)
    
    elif name=='resnet' and dataset=='imagenet':
        print("\n Fetching ResNet")
        print("Input size:", IMG_SIZE)
        net = ResNet18(num_classes=10, input_size=IMG_SIZE)

    elif name=='densenet' and dataset=='imagenet':
        net = DenseNet121(num_classes=10)
    
    elif name=='inception' and dataset=='imagenet':
        net = GoogLeNet(num_classes=10)
    
    elif name=='alexnet' and dataset=='imagenet':
        print("\n Fetching AlexNet")
        print("Input size:", IMG_SIZE)
        net = AlexNet(num_classes=10, input_size=IMG_SIZE)
    
    else:
        raise ValueError(f"{name} and {dataset} combination not valid")

    print("Trainable Params:", sum(p.numel() for p in net.parameters() if p.requires_grad), '\n')
    return net

def get_criterion(dataset):
    criterion = nn.CrossEntropyLoss()
    ''' Prepare criterion '''
    '''
    if dataset in ['cifar10', 'cifar10_gray', 'imagenet', 'fashion_mnist', 'svhn']:
        criterion = nn.CrossEntropyLoss()
    elif dataset in ['mnist', 'mnist_adversarial']:
        criterion = F.nll_loss
    ''' 
    return criterion 

def init_from_checkpoint(net):
    ''' Initialize from checkpoint'''
    print('==> Initializing  from fixed checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    
    checkpoint = torch.load('./checkpoint/' + args.net + '/' + args.net + '_' + args.dataset + '_ss' + args.iter + '/ckpt_trial_' + str(args.fixed_init) + '_epoch_50.t7')
    
    net.load_state_dict(checkpoint['net'])
    
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net, best_acc, start_epoch
