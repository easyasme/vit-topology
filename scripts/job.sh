#!/bin/sh
NETS="lenet" # lenetext alexnet resnet"

TRAIN=0 # train model; if 0, load model from checkpoint
BUILD_GRAPH=0 # build graph; if 0, load graph from binary file

TRIALS=1 # number of experiments that correspond to subsets of data; max is 29

N_EPOCHS=50
EPOCHS_TEST="1 10 20 30 40 50" # points where functional graph will be buit

UPPER_DIM=2 # must match config.py as well

## Train and compute topology for each dataset then create graphs
for net in $NETS
do
    for i in $(seq 0 "$TRIALS")
    do
        python main.py --net "$net" --dataset "imagenet" --trial 0 --lr 0.005  --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --graph_type functional --train "$TRAIN" --build_graph "$BUILD_GRAPH" --iter $i

        echo
        printf -- '-%.0s' $(seq 50)
        echo "\nVisualize Persistence Diagrams"
        printf -- '-%.0s' $(seq 50)
        echo

        for j in $(seq 1 "$UPPER_DIM")
        do
            echo "\nBetti $j"
            python visualize.py --trial 0 --net "$net" --dataset "imagenet" --epochs $(echo $EPOCHS_TEST) --dim $j --iter $i
        done
    done
done
