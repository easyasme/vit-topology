#!/bin/sh
DATASETS="imagenet" #cifar10_gray28 fashion_mnist svhn_gray28"
NETS="alexnet" #alexnet conv_x densenet inception resnet vgg"

N_EPOCHS=2
EPOCHS_TEST="1 2"

UPPER_DIM=2

## Train and compute topology for each dataset
for net in $NETS
do
    for i in {0..0}
    do
        python main.py --net "$net" --dataset "imagenet" --trial 0 --lr 0.0005  --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --graph_type functional --train 1 --build_graph 1 --iter $i

        for j in $(seq 1 "$UPPER_DIM")
        do
            python visualize.py --trial 0 --net "$net" --dataset "imagenet" --epochs $(echo $EPOCHS_TEST) --dim $j
        done
    done
done
