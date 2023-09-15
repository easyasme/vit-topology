import os
from .lenet import *
from .vgg import *
from .resnet import *
from .alexnet import *
from .densenet import *
from .inception import *
from .conv_x import * 

from config import IMG_SIZE

def get_model(name, dataset):
    if name == 'conv_2':
        net = Conv_2(num_classes=10)
    if name == 'conv_4':
        net = Conv_4(num_classes=10)
    if name == 'conv_6':
        net = Conv_6(num_classes=10)
    

    if name=='lenet' and dataset == 'imagenet':
        print("Fetching LeNet")
        print("Input size: ", IMG_SIZE)
        net = LeNet(num_channels=3, num_classes=10, input_size=IMG_SIZE)
    if name=='lenetext' and dataset=='imagenet':
        print("Fetching LeNetExt")
        print("Input size: ", IMG_SIZE)
        net = LeNetExt(n_channels=3, num_classes=10, input_size=IMG_SIZE)
    

    if name=='vgg' and dataset=='imagenet':
        net = VGG('VGG16', num_classes=10)
    

    if name=='resnet' and dataset=='imagenet':
        print("Fetching ResNet")
        print("Input size: ", IMG_SIZE)
        net = ResNet18(num_classes=10, input_size=IMG_SIZE)


    if name=='densenet' and dataset=='imagenet':
        net = DenseNet121(num_classes=10)
    

    if name=='inception' and dataset=='imagenet':
        net = GoogLeNet(num_classes=10)
    

    if name=='alexnet' and dataset=='imagenet':
        print("Fetching AlexNet")
        print("Input size: ", IMG_SIZE)
        net = AlexNet(num_classes=10, input_size=IMG_SIZE)


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
