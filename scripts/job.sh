#!/bin/sh
NETS="lenet" # conv_x densenet inception resnet vgg lenet lenetext"

TRIALS=1 # number of experiments that correspond to subsets of data; max is 29

N_EPOCHS=1
EPOCHS_TEST="1" # points where functional graph will be buit

UPPER_DIM=3 # must match config.py as well

## Train and compute topology for each dataset then create graphs
for net in $NETS
do
    for i in $(seq 0 "$TRIALS")
    do
        python main.py --net "$net" --dataset "imagenet" --trial 0 --lr 0.005  --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --graph_type functional --train 1 --build_graph 1 --iter $i
        
        echo
        printf -- '-%.0s' $(seq 50)
        echo "\nVisualize Persistence Diagrams"
        printf -- '-%.0s' $(seq 50)
        echo

        for j in $(seq 1 "$UPPER_DIM")
        do
            echo "\nBetti $j"
            python visualize.py --trial 0 --net "$net" --dataset "imagenet" --epochs $(echo $EPOCHS_TEST) --dim $j
        done
    done
done
