#!/bin/sh
DATASETS="cifar10" #cifar10_gray28 fashion_mnist svhn_gray28"
NETS="alexnet" #alexnet conv_x densenet inception resnet vgg"

N_EPOCHS=2 #50
EPOCHS_TEST="1 2" #10 20 30 40 50"

UPPER_DIM=2

## Train and compute topology for each dataset
for net in $NETS
do
    for dataset in $DATASETS
    do
        python main.py --net "$net" --dataset "$dataset" --trial 0 --lr 0.0005  --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --graph_type functional --train 1 --build_graph 1

        for i in $(seq 1 "$UPPER_DIM")
        do
            python visualize.py --trial 0 --net "$net" --dataset "$dataset" --epochs $(echo $EPOCHS_TEST) --dim $i
        done
    done
done
