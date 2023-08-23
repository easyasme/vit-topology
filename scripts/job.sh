#!/bin/sh
NETS="resnet" #alexnet conv_x densenet inception resnet vgg"

N_EPOCHS=50
EPOCHS_TEST="1 10 20 30 40 50"

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
