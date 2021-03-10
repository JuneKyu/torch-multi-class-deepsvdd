#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import wget
import zipfile
import pickle
import glob
import csv
from shutil import move
import numpy as np
from matplotlib import cm

from torch.utils.data import Subset
from PIL import Image, ImageMath
#  from torchvision import datasets

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization, divide_data_label

import torchvision.transforms as transforms

import pdb

class GTSRB_Dataset(TorchvisionDataset):
    def __init__(self, root:str, normal_class=[0]):
        super().__init__(root)

        self.normal_classes = tuple(normal_class)
        self.outlier_classes = list(range(0, 42))

        speed_limits = [0, 1, 2, 3, 4, 5, 7, 8]
        driving_instructions = [9, 10, 15, 16]
        warnings = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        directions = [33, 34, 35, 36, 37, 38, 39, 40]
        special_signs = [12, 13, 14, 17]
        regulations = [6, 32, 41, 42]
        total = speed_limits + driving_instructions + warnings + directions + special_signs + regulations
        scenario_classes = (speed_limits, driving_instructions, warnings, directions, special_signs, regulations)
        total.sort()

        normal_scenario = scenario_classes[normal_class[0]]

        for i in self.normal_classes:
            self.outlier_classes.remove(i)

        with open(os.path.join(self.root, 'gtsrb_min_max.pkl'), 'rb') as f:
            min_max = pickle.load(f)

        normal_mean = 0
        normal_std = 0
        for i in self.normal_classes:
            normal_mean += min_max[i][0]
            normal_std += min_max[i][1] - min_max[i][0]

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: global_contrast_normalization(x, scale='l1')),
            transforms.Normalize([normal_mean] * 3, [normal_std] * 3)
        ])

        target_transform = transforms.Lambda(
            lambda x: int(x in self.outlier_classes))

        train_set = MyGTSRB(root=self.root,
                            train=True,
                            download=True,
                            transform=transform,
                            target_transform=target_transform)

        train_idx_normal = get_target_label_idx(train_set.train_labels,
                                                self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyGTSRB(root=self.root,
                                train=False,
                                download=True,
                                transform=transform,
                                target_transform=target_transform)



class MyGTSRB(Dataset):
    def __init__(self, root, train, download, transform, target_transform):
        super(Dataset).__init__()
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self.train_data = []
        self.test_data = []

        if (os.path.exists(os.path.join(self.root, 'GTSRB')) == False):
            gtsrb_url = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370'
            gtsrb_train_url = os.path.join(gtsrb_url, 'GTSRB_Final_Training_Images.zip')
            gtsrb_test_url = os.path.join(gtsrb_url, 'GTSRB_Final_Test_Images.zip')
            gtsrb_test_gt_url = os.path.join(gtsrb_url, 'GTSRB_Final_Test_GT.zip')
            wget.download(gtsrb_train_url, out=self.root)
            wget.download(gtsrb_test_url, out=self.root)
            wget.download(gtsrb_test_gt_url, out=self.root)
            
            gtsrb_train_zip = zipfile.ZipFile(os.path.join(self.root, 'GTSRB_Final_Training_Images.zip'))
            gtsrb_test_zip = zipfile.ZipFile(os.path.join(self.root, 'GTSRB_Final_Test_Images.zip'))
            gtsrb_test_gt_zip = zipfile.ZipFile(os.path.join(self.root, 'GTSRB_Final_Test_GT.zip'))
            gtsrb_train_zip.extractall(root)
            gtsrb_test_zip.extractall(root)
            gtsrb_test_gt_zip.extractall(os.path.join(root, 'GTSRB'))
            os.remove(os.path.join(root, 'GTSRB_Final_Training_Images.zip'))
            os.remove(os.path.join(root, 'GTSRB_Final_Test_Images.zip'))
            os.remove(os.path.join(root, 'GTSRB_Final_Test_GT.zip'))
            divide_test_path(root)
            change_file_format(os.path.join(root, 'GTSRB/Final_Training/Images'))
            change_file_format(os.path.join(root, 'GTSRB/Final_Test/Images'))


        train_path = os.path.join(self.root, 'GTSRB/Final_Training/Images')
        test_path = os.path.join(self.root, 'GTSRB/Final_Test/Images')

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
            if d[0] == 0:
                continue
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
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

def divide_test_path(directory='../data'):

    test_annotations = get_test_labels(os.path.join(directory, 'GTSRB/GT-final_test.csv'))
    test_path = os.path.join(directory, 'GTSRB/Final_Test/Images')
    # make class dirs
    for i in range(43):
        os.mkdir(os.path.join(directory, 'GTSRB/Final_Test/Images/' + "{:05d}".format(i)))
    for annotation in test_annotations:
        path = os.path.join(test_path, annotation[0])
        dest = os.path.join(test_path, '{:05d}'.format(annotation[1]) + '/' + annotation[0])
        move(path, dest)


def get_test_labels(directory='../data/GTSRB/GT-final_test.csv'):
    annotations = []
    with open(directory) as f:
        reader = csv.reader(f, delimiter=';')
        next(reader) # skip header
        for row in reader:
            filename = row[0]
            label = int(row[7])
            annotations.append((filename, label))
    return annotations


def change_file_format(directory):
    # directory == Final_Training/Images
    classes = os.listdir(directory)
    for c in classes:
        if os.path.isdir(os.path.join(directory, c)):
            file_names = os.listdir(os.path.join(directory, c))
            for file_name in file_names:
                if file_name.split('.')[-1] == 'ppm':
                    file_path = directory + '/' + c + '/' + file_name
                    img = Image.open(file_path)
                    jpg_name = file_name.split('.')[0] + '.jpg'
                    dest_path = directory + '/' + c + '/' + jpg_name
                    os.remove(file_path)
                    img.save(dest_path)
