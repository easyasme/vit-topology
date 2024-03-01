import argparse
import os

from config import SAVE_PATH

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--build_graph', default=1, type=int)
    parser.add_argument('--net', help='Specify deep network architecture.')
    parser.add_argument('--dataset', help='Specify dataset (e.g. mnist, cifar10, imagenet)')
    parser.add_argument('--n_epochs_train', default='10', help='Number of epochs to train.')
    parser.add_argument('--lr', default='0.001', help='Specify learning rate for training.')
    parser.add_argument('--optimizer', default='adabelief', help='Specify optimizer for training. (e.g. adabelief or adam')
    parser.add_argument('--epochs_test', default='1 5 10', help='Epochs for which you want to build graph.')
    parser.add_argument('--thresholds', default='0.5 1.0', help='Defining thresholds range in the form \'start stop\' ')
    parser.add_argument('--reduction', default='pca', type=str, help='Reductions: pca, tsne, umap, or none.')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--verbose', default=0, type=int)

    args = parser.parse_args()

    def visible_print(message):
        ''' Visible print'''
        print('')
        print(50*'-')
        print(message)
        print(50*'-')
        print('')

    ONAME = args.net + '_' + args.dataset + '_ss' + str(args.iter)
    SAVE_DIR = os.path.join(SAVE_PATH, ONAME)

    # if args.reduction is not None:
    #     from dask_cuda import LocalCUDACluster
    #     import logging as log

    #     INFINIBAND = True if os.uname()[1].split('-')[0].__eq__('dw') else False
    #     print(f'\n\nInfiniband: {INFINIBAND}')

    #     CLUSTER = LocalCUDACluster(
    #         protocol="ucx",
    #         n_workers=3, # num of gpus
    #         threads_per_worker=1, # num of threads per gpu
    #         memory_limit='auto',
    #         enable_tcp_over_ucx=True,
    #         enable_nvlink=True,
    #         enable_infiniband=INFINIBAND,
    #         enable_rdmacm=INFINIBAND,
    #         silence_logs=log.ERROR, # only log errors to IO
    #         rmm_pool_size=None, # fraction of GPU memory to allocate for RMM pool
    #         rmm_maximum_pool_size=None,
    #         device_memory_limit='80GiB', # limit at which to spill to cpu memory
    #     )
    #     # print(CLUSTER.logs())
    #     # print(f'Memory limit: {CLUSTER.memory_limit / 1e9:.2f} GB')
    #     # exit()

    if args.train:
        visible_print('Training network')
        os.system('python ./train.py --net '+args.net+' --dataset '+args.dataset+' --epochs '+args.n_epochs_train+' --lr '+args.lr+' --iter '+str(args.iter)+' --chkpt_epochs '+args.epochs_test+' --optimizer '+args.optimizer)

    if args.build_graph:
        visible_print('Building graph')
        os.system('python ./build_graph_functional.py --save_dir '+SAVE_DIR+' --net '+args.net+' --dataset '+args.dataset+' --chkpt_epochs '+args.epochs_test+' --iter '+str(args.iter)+' --verbose '+str(args.verbose)+' --reduction '+args.reduction)#+' --cluster '+CLUSTER.scheduler_address)

    # CLUSTER.close()