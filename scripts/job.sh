#!/bin/sh
DATASETS="mnist" #cifar10_gray28 fashion_mnist svhn_gray28"
NETS="lenet" #alexnet conv_x densenet inception resnet vgg"

<<<<<<< HEAD
N_EPOCHS=2 #50
EPOCHS_TEST="1 2" #10 20 30 40 50"

UPPER_DIM=4
=======
N_EPOCHS=50
EPOCHS_TEST="1 10 20 30 40 50"

UPPER_DIM=3
>>>>>>> 8f10f5c19fc7e2c189e9fce0a226c6aa1a4e00da

## Train and compute topology for each dataset
for dataset in $DATASETS
do
    python main.py --net lenet --dataset "$dataset" --trial 0 --lr 0.0005  --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --graph_type functional --train 1 --build_graph 1

    for i in $(seq 1 "$UPPER_DIM")
    do
        python visualize.py --trial 0 --net lenet --dataset "$dataset" --epochs $(echo $EPOCHS_TEST) --dim $i
    done
done
