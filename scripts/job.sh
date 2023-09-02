#!/bin/sh
NETS="resnet" # conv_x densenet inception resnet vgg lenet lenetext"

N_EPOCHS=60
EPOCHS_TEST="1 10 20 40 60"

UPPER_DIM=3

## Train and compute topology for each dataset then create graphs
for net in $NETS
do
    for i in $(seq 1 1)
    do
        python main.py --net "$net" --dataset "imagenet" --trial 0 --lr 0.005  --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --graph_type functional --train 1 --build_graph 1 --iter $i
        printf "%0.s-" {1..50}
        echo "Visualize Persistence Diagrams"
        printf "%0.s-" {1..50}

        for j in $(seq 1 "$UPPER_DIM")
        do
            echo "Betti $j"
            python visualize.py --trial 0 --net "$net" --dataset "imagenet" --epochs $(echo $EPOCHS_TEST) --dim $j
            echo
        done
    done
done
