import os

from config import IMG_SIZE

from .alexnet import *
from .conv_x import *
from .densenet import *
from .fcnet import *
from .inception import *
from .lenet import *
from .resnet import *
from .vgg import *


def get_model(name, dataset):
    if name == 'conv_2':
        print("\n Fetching Conv_2")
        print("Input size:", IMG_SIZE)
        net = Conv_2(num_classes=10)
    elif name == 'conv_4':
        print("\n Fetching Conv_4")
        print("Input size:", IMG_SIZE)
        net = Conv_4(num_classes=10)
    elif name == 'conv_6':
        print("\n Fetching Conv_6")
        print("Input size:", IMG_SIZE)
        net = Conv_6(num_classes=10)

    elif name == 'fcnet':
        print("\n Fetching FCNet")
        print("Input size: ", 3, '\n')
        net = FCNet(input_size=3, num_classes=3)
    
    elif name=='lenet' and dataset == 'mnist':
        print("\n Fetching LeNet")
        print("Input size: ", 28, '\n')
        net = LeNet(num_channels=1, num_classes=10, input_size=28)
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

    elif name=='vgg' and dataset=='mnist':
        print("\n Fetching VGG9")
        print("Input size:", 28, '\n')
        net = VGG('VGG9', img_size=28, num_classes=10)
    elif name=='vgg' and dataset=='imagenet':
        print("\n Fetching VGG16")
        print("Input size:", IMG_SIZE)
        net = VGG('VGG16', img_size=IMG_SIZE, num_classes=10)
    
    elif name=='resnet' and dataset=='imagenet':
        print("\n Fetching ResNet")
        print("Input size:", IMG_SIZE)
        net = ResNet18(num_classes=10, input_size=IMG_SIZE)

    elif name=='densenet' and dataset=='imagenet':
        print("\n Fetching DenseNet121")
        print("Input size:", IMG_SIZE)
        net = DenseNet121(num_classes=10)
    
    elif name=='inception' and dataset=='imagenet':
        print("\n Fetching InceptionV3")
        print("Input size:", IMG_SIZE)
        net = GoogLeNet(num_classes=10)
    
    elif name=='alexnet' and dataset=='imagenet':
        print("\n Fetching AlexNet")
        print("Input size:", IMG_SIZE)
        net = AlexNet(num_classes=10, input_size=IMG_SIZE)
    
    else:
        raise ValueError(f"{name} and {dataset} combination not valid")

    print("Trainable Params:", sum(p.numel() for p in net.parameters() if p.requires_grad), '\n')
    return net

def init_from_checkpoint(net, optimizer, args):
    ''' Initialize from checkpoint'''
    print('==> Initializing  from fixed checkpoint..')
    assert os.path.isdir('./checkpoint'), 'Error: no checkpoint directory found!'
    
    if args.dataset == 'imagenet':
        checkpoint = torch.load(f'./checkpoint/{args.net}/{args.net}_{args.dataset}_ss{args.iter}/ckpt_epoch_{args.resume_epoch}.pt')
    else:
        checkpoint = torch.load(f'./checkpoint/{args.net}/{args.net}_{args.dataset}/ckpt_epoch_{args.resume_epoch}.pt')
    
    keys = checkpoint.keys()
    if 'net' in keys:
        net.load_state_dict(checkpoint['net'])
    if 'optimizer' in keys:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'loss_tr' in keys:
        loss_tr = checkpoint['loss_tr']
    if 'loss_te' in keys:
        loss_te = checkpoint['loss_te']
    if 'acc_tr' in keys:
        acc_tr = checkpoint['acc_tr']
    if 'acc_te' in keys:
        acc_te = checkpoint['acc_te']
    if 'epoch' in keys:
        epoch = checkpoint['epoch']

    del keys, checkpoint

    return net, optimizer, loss_tr, loss_te, acc_tr, acc_te, epoch
