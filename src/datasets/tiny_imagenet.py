import os
import wget
import zipfile
import pickle
import glob
from shutil import move, rmtree, copytree

from torch.utils.data import Subset
from PIL import Image
#  from torchvision import datasets

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization, divide_data_label

import torchvision.transforms as transforms

import pdb

class TINY_Imagenet_Dataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class=[5]):
        super().__init__(root)

        self.normal_classes = tuple(normal_class)
        self.outlier_classes = list(range(0, 200))

        train_classes = []
        with open(os.path.join(root, 'tiny-imagenet-200/tiny_imagenet_classes.txt'), 'r') as f:
            item = f.read()
            train_classes = item.split('\n')[:-1]

        animal = [66, 90, 134, 139, 148, 180, 182, 191]
        insect = [13, 31, 92, 123, 165, 177, 196, 199]
        instruments = [16, 17, 18, 72, 74, 116, 128, 197]
        structure = [48, 58, 69, 96, 122, 151, 157, 178]
        transportation = [0, 22, 23, 26, 46, 47, 156, 169]
        total_index = animal + insect + instruments + structure + transportation

        #  animal = change_index(train_classes, animal)
        #  insect = change_index(train_classes, insect)
        #  instruments = change_index(train_classes, instruments)
        #  structure = change_index(train_classes, structure)
        #  transportation = change_index(train_classes, transportation)

        total = animal + insect + instruments + structure + transportation
        scenario_classes = (animal, insect, instruments, structure, transportation)
        total.sort()
        self.outlier_classes = (total)

        normal_scenario = scenario_classes[normal_class[0]]

        for i in normal_scenario:
            self.outlier_classes.remove(i)

        with open(os.path.join(self.root, 'tiny_imagenet_200_min_max.pkl'), 'rb') as f:
            min_max = pickle.load(f)

        normal_mean = 0
        normal_std = 0
        for i in self.normal_classes:
            normal_mean += min_max[i][0]
            normal_std += min_max[i][1] - min_max[i][0]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([normal_mean] * 3, [normal_std] * 3)
        ])

        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))

        train_set = MyTinyImagenet(root=self.root,
                                   train=True,
                                   download=True,
                                   transform=transform,
                                   target_transform=target_transform)

        train_idx_normal = get_target_label_idx(train_set.train_labels,
                                                #  self.normal_classes)
                                                normal_scenario)

        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyTinyImagenet(root=self.root,
                                       train=False,
                                       download=True,
                                       transform=transform,
                                       target_transform=target_transform)


def val_dataset_labeling(val_path):
    val_dict = {}
    with open(os.path.join(val_path, 'val_annotations.txt'), 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob(os.path.join(val_path, 'images/*'))
    paths[0].split('/')[-1]
    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        if not os.path.exists(os.path.join(val_path, str(folder))):
            os.mkdir(os.path.join(val_path, str(folder)))

    for path in paths:
        file = path.split('/')[-1]
        folder = val_dict[file]
        dest = os.path.join(val_path, str(folder), str(file))
        move(path,dest)

    os.remove(os.path.join(val_path, 'val_annotations.txt'))
    os.rmdir(os.path.join(val_path, 'images'))


def change_index(classes, indexes):
    
    class_indexes = []
    for i in indexes:
        class_indexes.append(classes[i])

    return class_indexes


class MyTinyImagenet(Dataset):
    def __init__(self, root, train, download, transform, target_transform):
        super(Dataset).__init__()
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self.train_data = []
        self.test_data = []

        if (os.path.exists(os.path.join(self.root, 'tiny-imagenet-200')) == False):
            tiny_imagenet_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            wget.download(tiny_imagenet_url, out=self.root)
            tiny_imagenet_zip = zipfile.ZipFile(os.path.join(self.root, 'tiny-imagenet-200.zip'))
            tiny_imagenet_zip.extractall(root)
            os.remove(os.path.join(root, 'tiny-imagenet-200.zip'))
            val_dataset_labeling(os.path.join(self.root, 'tiny-imagenet-200/val'))

        train_path = os.path.join(self.root, 'tiny-imagenet-200/train')
        test_path = os.path.join(self.root, 'tiny-imagenet-200/val')

        train_d = ImageFolder(train_path)
        test_d = ImageFolder(test_path)

        self.train_data = []
        self.train_labels = []
        for d in train_d:
            self.train_data.append(d[0])
            self.train_labels.append(d[1])
        self.test_data = []
        self.test_labels = []
        for d in test_d:
            self.test_data.append(d[0])
            self.test_labels.append(d[1])

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
            #  img, target = self.train_data[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
            #  img, target = self.test_data[index]

        #  img = Image.forarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
