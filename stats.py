from config import SUBSETS_LIST
from loaders import *

img_path_32 = '/home/trogdent/imagenet_data/train_32'
img_path_64 = '/home/trogdent/imagenet_data/train_64'

labels_path = './data/map_clsloc.txt'

batch_size = 32

stats = []
for i, subset in enumerate(SUBSETS_LIST):
    print("Subset: ", subset)
    
    transform = get_transform(train=True, crop=False, hflip=False, vflip=False, color_dis=False, blur=False, resize=False)
    dataset = CustomImageNet(img_path_64, labels_path, verbose=True, subset=subset, transform=transform, grayscale=False, iter=i)
    
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, drop_last=True,  worker_init_fn=seed_worker)

    stats.append(calc_mean_std(data_loader)) # pixel-wise mean and std
    

print("Means: ", stats[:, 0])
print("Stds: ", stats[:, 1])
