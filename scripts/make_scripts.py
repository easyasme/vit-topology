import argparse
import os


def main(pars):
    args = pars.parse_args()

    starts_stops = ['0 9', '10 19', '20 29']
    for ss in starts_stops:
        job_name = f'{args.net} {args.dataset}'
        job_name += f' {ss.split()[0]}-{ss.split()[1]}' if args.dataset == 'imagenet' else ''
        job_name += f' {args.reduction}' if args.reduction else ''
        job_name += f' {args.metric}' if args.metric else ''

        print(f'\n ==> Job name: {job_name} \n')

        pth = f'{args.reduction}' if args.reduction else ''
        pth = os.path.join(pth, args.metric if args.metric else '') 
        SAVE_DIR = os.path.join(args.save_dir, pth)

        print(f'\n ==> Save directory: {SAVE_DIR} \n')

        cmd = f"python ../../scriptor.py --net {args.net} --dataset {args.dataset} --data_subset '{ss}' --subset {args.subset} --train {args.train} --build_graph {args.build_graph} --post_process {args.post_process} --n_epochs_train {args.n_epochs_train} --lr {args.lr} --optimizer {args.optimizer} --epochs_test '{args.epochs_test}' --verbose {args.verbose} --time {args.time} --ntasks {args.ntasks} --mem {args.mem} --nodes {args.nodes} --gpus {args.gpus} --qos {args.qos} --user_email {args.user_email} --job_name '{job_name}' --save_dir '{SAVE_DIR}'"
        cmd += f' --reduction {args.reduction}' if args.reduction else ''
        cmd += f' --metric {args.metric}' if args.metric else ''

        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build and run SLURM scripts')
    required = parser.add_argument_group('required arguments')

    # Train and build graph parameters
    required.add_argument('--net', help='Specify deep network architecture.', required=True)
    required.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)', required=True)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--build_graph', default=1, type=int)
    parser.add_argument('--post_process', default=1, type=int)
    parser.add_argument('--n_epochs_train', default='50', help='Number of epochs to train.')
    parser.add_argument('--lr', default='0.001', help='Specify learning rate for training.')
    parser.add_argument('--optimizer', default='adabelief', help='Define training optimizer: "adabelief" or "adam")')
    parser.add_argument('--epochs_test', default='0 4 8 20 30 40 50', help='Epochs for which you want to build graph.')
    parser.add_argument('--subset', default=500, type=int, help='Subset size for building graph.')
    parser.add_argument('--metric', default=None, type=str, help='Distance metric: "spearman", "dcor", or callable.')
    parser.add_argument('--reduction', default=None, type=str, help='Reductions: pca, umap, or none.')
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--save_dir', default='./results/', help='Directory to save results.')

    # SLURM parameters
    parser.add_argument('--time', default='24:00:00', help="Walltime in the form 'hh:mm:ss'")
    parser.add_argument('--ntasks', default=120, type=int, help='Number of processor cores (i.e. tasks)')
    parser.add_argument('--mem', default='750G', help='Total CPU memory')
    parser.add_argument('--nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--gpus', default=1, type=int, help='Number of GPUs')
    parser.add_argument('--qos', default='cs', help='Quality of service')
    parser.add_argument('--user_email', default='name@institute.edu', type=str, help='Email address')
    
    main(parser)