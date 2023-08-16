import os
from .lenet import *
from .vgg import *
from .resnet import *
from .alexnet import *
from .densenet import *
from .inception import *
from .conv_x import * 

num_classes={'imagenet': 10}

def get_model(name, dataset):
    if name == 'conv_2':
        net = Conv_2(num_classes=10)
    if name == 'conv_4':
        net = Conv_4(num_classes=10)
    if name == 'conv_6':
        net = Conv_6(num_classes=10)
    
    if name=='lenet' and dataset == 'imagenet':
        net = LeNet(num_channels=3, num_classes=10)
    
    if name=='lenet32' and dataset == 'imagenet':
        net = LeNet(num_channels=3, num_classes=10, input_size=32)
    
    if name=='lenetext' and dataset=='imagenet':
        net = LeNetExt(n_channels=3, num_classes=10)
    
    if name=='vgg' and dataset=='imagenet':
        net = VGG('VGG16', num_classes=10)
    
    if name=='resnet' and dataset=='imagenet':
        net = ResNet18(num_classes=10)
       
    if name=='densenet' and dataset=='imagenet':
        net = DenseNet121(num_classes=10)
    
    if name=='inception' and dataset=='imagenet':
        net = GoogLeNet(num_classes=10)
    
    if name=='alexnet' and dataset=='imagenet':
        net = AlexNet(num_classes=10)

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
    
    checkpoint = torch.load('./checkpoint/' + args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.fixed_init) + '_epoch_50.t7')
    
    net.load_state_dict(checkpoint['net'])
    
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    return net, best_acc, start_epoch
