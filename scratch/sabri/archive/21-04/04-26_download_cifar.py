root_dir = "/home/common/datasets"

from torchvision.datasets import CIFAR10

dataset = CIFAR10(download=True, root=root_dir)
