from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .tiny_imagenet import TINY_Imagenet_Dataset
from .gtsrb import GTSRB_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'tiny_imagenet', 'gtsrb')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'tiny_imagenet':
        dataset = TINY_Imagenet_Dataset(root=data_path, normal_class=normal_class)
        
    if dataset_name == 'gtsrb':
        dataset = GTSRB_Dataset(root=data_path, normal_class=normal_class)

    return dataset
