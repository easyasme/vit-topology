''' SLURM batch job script generator '''

import argparse
import os

parser = argparse.ArgumentParser(description='Build and run SLURM scripts')
required = parser.add_argument_group('required arguments')

# Train and build graph parameters
required.add_argument('--net', help='Specify deep network architecture.', required=True)
required.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)', required=True)
parser.add_argument('--train', default=1, type=int)
parser.add_argument('--build_graph', default=1, type=int)
parser.add_argument('--data_subset', default='0 10', help='Specify data subset in the form \'start stop\'')
parser.add_argument('--subset', default=500, type=int, help='Subset size for building graph.')
parser.add_argument('--metric', default=None, type=str, help='Distance metric: "spearman", "dcor", or callable.')
parser.add_argument('--n_epochs_train', default='50', help='Number of epochs to train.')
parser.add_argument('--lr', default='0.001', help='Specify learning rate for training.')
parser.add_argument('--optimizer', default='adabelief', help='Specify optimizer for training. (e.g. adabelief or adam)')
parser.add_argument('--epochs_test', default='0 4 8 20 30 40 50', help='Epochs for which you want to build graph.')
parser.add_argument('--reduction', default='pca', type=str, help='Reductions: pca, umap, or none.')
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--save_dir', default='./results/', help='Directory to save results.')

# SLURM parameters
parser.add_argument('--time', default='24:00:00', help="Walltime in the form hh:mm:ss")
parser.add_argument('--ntasks', default=120, type=int, help='Number of processor cores (i.e. tasks)')
parser.add_argument('--mem', default='750G', help='Total CPU memory')
parser.add_argument('--nodes', default=1, type=int, help='Number of nodes')
parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs')
parser.add_argument('--qos', default='cs', help='Quality of service')
parser.add_argument('--user_email', default='name@institute.edu', type=str, help='Email address')
parser.add_argument('--job_name', default='train and build', type=str, help='Job name')

args = parser.parse_args()

start = int(args.data_subset.split()[0])
stop = int(args.data_subset.split()[1])


if args.dataset.__eq__('mnist'):
    FILENAME = f'job_{args.net}_{args.dataset}.sh'
else:
    FILENAME = f'job_{args.net}_{args.dataset}_{start}-{stop}.sh'

print(f'Generating job script: {FILENAME}\n')
if (args.reduction is not None) and (args.metric is None):
    with open(FILENAME, 'w') as f:
        f.write(f'''\
    #!/bin/sh

    NET="{args.net}" # lenet lenetext alexnet resnet densenet vgg inception
    DATASET="{args.dataset}" # mnist imagenet etc.

    TRAIN={args.train} # train model; if 0, load model from checkpoint
    BUILD_GRAPH={args.build_graph} # build graph; if 0, load graph from binary file

    OPTIMIZER="{args.optimizer}" # adam, adabelief; if '' then use sgd

    RED="{args.reduction}" # pca or umap
                
    VERBOSE={args.verbose} # verbose level

    START={start} # start index of experiment
    STOP={stop} # number of experiments that correspond to subsets of data; max is 29

    SAVE_DIR="{args.save_dir}" # directory to save results

    # training params
    N_EPOCHS={args.n_epochs_train} # number of epochs to train
    EPOCHS_TEST='{args.epochs_test}' # points where functional graph will be buit
    LR={args.lr} # learning rate

    ## Train and compute topology for each dataset then create graphs
    echo
    printf -- '-%.0s' $(seq 50)
    echo
    echo "Training $NET on $DATASET"
    echo "Epochs: $N_EPOCHS"
    echo "Epochs to build graph: $EPOCHS_TEST"
    echo "Learning rate: $LR"
    printf -- '-%.0s' $(seq 50)

    if [ $DATASET == "mnist" ]
    then
        python main.py --net "$NET" --dataset "$DATASET" --lr "$LR" --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --train "$TRAIN" --build_graph "$BUILD_GRAPH" --optimizer "$OPTIMIZER" --reduction "$RED" --verbose "$VERBOSE" --save_dir "$SAVE_DIR"
    else
        for i in $(seq "$START" "$STOP")
        do
            echo
            echo "Subset $i"
            python main.py --net "$NET" --dataset "$DATASET" --lr "$LR" --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --train "$TRAIN" --build_graph "$BUILD_GRAPH" --optimizer "$OPTIMIZER" --reduction "$RED" --iter $i --verbose "$VERBOSE" --save_dir "$SAVE_DIR"
        done
    fi
    ''')
elif (args.metric is not None) and (args.reduction is None):
    with open(FILENAME, 'w') as f:
        f.write(f'''\
    #!/bin/sh

    NET="{args.net}" # lenet lenetext alexnet resnet densenet vgg inception
    DATASET="{args.dataset}" # mnist imagenet etc.

    TRAIN={args.train} # train model; if 0, load model from checkpoint
    BUILD_GRAPH={args.build_graph} # build graph; if 0, load graph from binary file

    OPTIMIZER="{args.optimizer}" # adam, adabelief; if '' then use sgd

    METRIC="{args.metric}" # distance metric: spearman, dcorr, or callable
                
    VERBOSE={args.verbose} # verbose level

    START={start} # start index of experiment
    STOP={stop} # number of experiments that correspond to subsets of data; max is 29

    SAVE_DIR="{args.save_dir}" # directory to save results

    # training params
    N_EPOCHS={args.n_epochs_train} # number of epochs to train
    EPOCHS_TEST='{args.epochs_test}' # points where functional graph will be buit
    LR={args.lr} # learning rate

    ## Train and compute topology for each dataset then create graphs
    echo
    printf -- '-%.0s' $(seq 50)
    echo
    echo "Training $NET on $DATASET"
    echo "Epochs: $N_EPOCHS"
    echo "Epochs to build graph: $EPOCHS_TEST"
    echo "Learning rate: $LR"
    printf -- '-%.0s' $(seq 50)

    if [ $DATASET == "mnist" ]
    then
        python main.py --net "$NET" --dataset "$DATASET" --lr "$LR" --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --train "$TRAIN" --build_graph "$BUILD_GRAPH" --optimizer "$OPTIMIZER" --metric "$METRIC" --verbose "$VERBOSE" --save_dir "$SAVE_DIR"
    else
        for i in $(seq "$START" "$STOP")
        do
            echo
            echo "Subset $i"
            python main.py --net "$NET" --dataset "$DATASET" --lr "$LR" --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --train "$TRAIN" --build_graph "$BUILD_GRAPH" --optimizer "$OPTIMIZER" --metric "$METRIC" --iter $i --verbose "$VERBOSE" --save_dir "$SAVE_DIR"
        done
    fi
    ''')
elif (args.metric is None) and (args.reduction is None):
    with open(FILENAME, 'w') as f:
        f.write(f'''\
    #!/bin/sh

    NET="{args.net}" # lenet lenetext alexnet resnet densenet vgg inception
    DATASET="{args.dataset}" # mnist imagenet etc.

    TRAIN={args.train} # train model; if 0, load model from checkpoint
    BUILD_GRAPH={args.build_graph} # build graph; if 0, load graph from binary file

    OPTIMIZER="{args.optimizer}" # adam, adabelief; if '' then use sgd
                
    VERBOSE={args.verbose} # verbose level

    START={start} # start index of experiment
    STOP={stop} # number of experiments that correspond to subsets of data; max is 29

    SAVE_DIR="{args.save_dir}" # directory to save results

    # training params
    N_EPOCHS={args.n_epochs_train} # number of epochs to train
    EPOCHS_TEST='{args.epochs_test}' # points where functional graph will be buit
    LR={args.lr} # learning rate

    ## Train and compute topology for each dataset then create graphs
    echo
    printf -- '-%.0s' $(seq 50)
    echo
    echo "Training $NET on $DATASET"
    echo "Epochs: $N_EPOCHS"
    echo "Epochs to build graph: $EPOCHS_TEST"
    echo "Learning rate: $LR"
    printf -- '-%.0s' $(seq 50)

    if [ $DATASET == "mnist" ]
    then
        python main.py --net "$NET" --dataset "$DATASET" --lr "$LR" --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --train "$TRAIN" --build_graph "$BUILD_GRAPH" --optimizer "$OPTIMIZER" --verbose "$VERBOSE" --save_dir "$SAVE_DIR"
    else
        for i in $(seq "$START" "$STOP")
        do
            echo
            echo "Subset $i"
            python main.py --net "$NET" --dataset "$DATASET" --lr "$LR" --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --train "$TRAIN" --build_graph "$BUILD_GRAPH" --optimizer "$OPTIMIZER" --iter $i --verbose "$VERBOSE" --save_dir "$SAVE_DIR"
        done
    fi
    ''')
else:
    with open(FILENAME, 'w') as f:
        f.write(f'''\
    #!/bin/sh

    NET="{args.net}" # lenet lenetext alexnet resnet densenet vgg inception
    DATASET="{args.dataset}" # mnist imagenet etc.

    TRAIN={args.train} # train model; if 0, load model from checkpoint
    BUILD_GRAPH={args.build_graph} # build graph; if 0, load graph from binary file

    OPTIMIZER="{args.optimizer}" # adam, adabelief; if '' then use sgd

    RED="{args.reduction}" # pca or umap

    METRIC="{args.metric}" # distance metric: spearman, dcorr, or callable
                
    VERBOSE={args.verbose} # verbose level

    START={start} # start index of experiment
    STOP={stop} # number of experiments that correspond to subsets of data; max is 29

    SAVE_DIR="{args.save_dir}" # directory to save results

    # training params
    N_EPOCHS={args.n_epochs_train} # number of epochs to train
    EPOCHS_TEST='{args.epochs_test}' # points where functional graph will be buit
    LR={args.lr} # learning rate

    ## Train and compute topology for each dataset then create graphs
    echo
    printf -- '-%.0s' $(seq 50)
    echo
    echo "Training $NET on $DATASET"
    echo "Epochs: $N_EPOCHS"
    echo "Epochs to build graph: $EPOCHS_TEST"
    echo "Learning rate: $LR"
    printf -- '-%.0s' $(seq 50)

    if [ $DATASET == "mnist" ]
    then
        python main.py --net "$NET" --dataset "$DATASET" --lr "$LR" --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --train "$TRAIN" --build_graph "$BUILD_GRAPH" --optimizer "$OPTIMIZER" --reduction "$RED" --metric "$METRIC" --verbose "$VERBOSE" --save_dir "$SAVE_DIR"
    else
        for i in $(seq "$START" "$STOP")
        do
            echo
            echo "Subset $i"
            python main.py --net "$NET" --dataset "$DATASET" --lr "$LR" --n_epochs_train "$N_EPOCHS" --epochs_test "$EPOCHS_TEST" --train "$TRAIN" --build_graph "$BUILD_GRAPH" --optimizer "$OPTIMIZER" --reduction "$RED" --metric "$METRIC" --iter $i --verbose "$VERBOSE" --save_dir "$SAVE_DIR"
        done
    fi
    ''')

if args.dataset.__eq__('mnist'):
    BATCH_FILE = f'jobscript_{args.net}_{args.dataset}'
else:
    BATCH_FILE = f'jobscript_{args.net}_{args.dataset}_{start}-{stop}'

with open(BATCH_FILE, 'w') as f:
    f.write(f'''\
#!/bin/bash --login

#SBATCH --time={args.time}   # walltime
#SBATCH --ntasks-per-node={args.ntasks}   # number of processor cores (i.e. tasks)
#SBATCH --mem={args.mem}   # total CPU memory
#SBATCH --nodes={args.nodes}   # num nodes
#SBATCH --gpus={args.gpus}
#SBATCH --qos={args.qos}
#SBATCH --mail-user={args.user_email}   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -J "{args.job_name}"   # job name

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_pml="ucx" 
export OMPI_MCA_osc="ucx"
export UCX_MEMTYPE_CACHE=n 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/topo_gph/lib

mamba activate topo_gph
cd ~/compute/qual/dnn-topology

. scripts/{args.reduction}/{args.metric}/{FILENAME}
''')

print(f'Batch job script generated: {BATCH_FILE}')
print(f'Job script generated: {FILENAME}\n')
print('Run the job script with the following command from the dnn-topology directory:')
print(f'sbatch scripts/{BATCH_FILE}\n')
