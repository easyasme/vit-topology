import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

other_transform = transforms.Normalize((0.5), (0.5))

img = torch.randn(3, 28, 28)
print(type(transform(img)))

print("Before: ", transform)

plt.imshow(transform(img).permute(1, 2, 0))

transform.transforms.insert(4, other_transform)

print("After: ", transform)

plt.imshow(transform(img).permute(1, 2, 0))