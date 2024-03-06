import argparse
import os


def main(pars):
    args = pars.parse_args()

    starts_stops = ['0 9', '10 19', '20 29']
    for subset in starts_stops:
        job_name = f'{args.net}_{args.dataset} {subset.split()[0]}-{subset.split()[1]}'

        os.system(f"python scriptor.py --net {args.net} --dataset {args.dataset} --data_subset '{subset}' --train {args.train} --build_graph {args.build_graph} --n_epochs_train {args.n_epochs_train} --lr {args.lr} --optimizer {args.optimizer} --epochs_test '{args.epochs_test}' --reduction {args.reduction} --verbose {args.verbose} --time {args.time} --ntasks {args.ntasks} --mem {args.mem} --nodes {args.nodes} --gpus {args.gpus} --qos {args.qos} --user_email {args.user_email} --job_name '{job_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build and run SLURM scripts')
    required = parser.add_argument_group('required arguments')

    # Train and build graph parameters
    required.add_argument('--net', help='Specify deep network architecture.', required=True)
    required.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)', required=True)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--build_graph', default=1, type=int)
    parser.add_argument('--n_epochs_train', default='50', help='Number of epochs to train.')
    parser.add_argument('--lr', default='0.001', help='Specify learning rate for training.')
    parser.add_argument('--optimizer', default='adabelief', help='Specify optimizer for training. (e.g. adabelief or adam)')
    parser.add_argument('--epochs_test', default='0 4 8 20 30 40 50', help='Epochs for which you want to build graph.')
    parser.add_argument('--reduction', default='pca', type=str, help='Reductions: pca, umap, or none.')
    parser.add_argument('--verbose', default=0, type=int)

    # SLURM parameters
    parser.add_argument('--time', default='24:00:00', help="Walltime in the form 'hh:mm:ss'")
    parser.add_argument('--ntasks', default=120, type=int, help='Number of processor cores (i.e. tasks)')
    parser.add_argument('--mem', default='750G', help='Total CPU memory')
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs')
    parser.add_argument('--qos', default='cs', help='Quality of service')
    parser.add_argument('--user_email', default='name@institute.edu', type=str, help='Email address')
    
    main(parser)