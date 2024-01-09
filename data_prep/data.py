import os
import PIL
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict

from data_prep.common import create_lda_partitions
from constant import *


def get_dataset(args):
    if args.dataset == CIFAR100:
        return get_cifar100(args)
    elif args.dataset == tinyImageNet:
        return get_tiny(args)
    else:
        raise NotImplementedError


def get_cifar100(args):
    args.num_classes = 100
    normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                     (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    args.img_size = 32
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    train_dataset = datasets.CIFAR100(root=args.path, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=args.path, train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset


def get_tiny(args):
    args.num_classes = 200

    def parse_classes(file):
        classes = []
        filenames = []
        with open(file) as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        for x in range(len(lines)):
            tokens = lines[x].split()
            classes.append(tokens[1])
            filenames.append(tokens[0])
        return filenames, classes

    class TinyImageNetDataset(torch.utils.data.Dataset):
        """Dataset wrapping images and ground truths."""
        def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
            self.img_path = img_path
            self.transform = transform
            self.gt_path = gt_path
            self.class_to_idx = class_to_idx
            self.classidx = []
            self.imgs, self.classnames = parse_classes(gt_path)
            for classname in self.classnames:
                self.classidx.append(self.class_to_idx[classname])
            self.targets = self.classidx

        def __getitem__(self, index):
            img = None
            with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
            y = self.classidx[index]
            return img, y

        def __len__(self):
            return len(self.imgs)

    data_path = args.path
    args.img_size = 64
    train_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'tiny-imagenet-200', 'train'),
        transform=transforms.Compose(
            [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    )

    test_dataset = TinyImageNetDataset(
        img_path=os.path.join(data_path, 'tiny-imagenet-200', 'val', 'images'),
        gt_path=os.path.join(data_path, 'tiny-imagenet-200', 'val', 'val_annotations.txt'),
        class_to_idx=train_dataset.class_to_idx.copy(),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    )
    return train_dataset, test_dataset


class CL_dataset():
    def __init__(self, args):
        self.args = args
        self.name = args.dataset
        self.train_dataset, self.test_dataset = get_dataset(self.args)
        self.classes = np.arange(len(np.unique(self.train_dataset.targets)))
        self.train_ds, self.cl_test_loaders, total_test = [], [], []
        self.full_test_loaders, current_train, current_test = [], [], []
        self.n_classes_per_task = len(self.classes) // self.args.n_tasks
        for i, label in enumerate(self.classes):
            current_train.extend(np.where(self.train_dataset.targets == label)[0].tolist())
            current_test.extend(np.where(self.test_dataset.targets == label)[0].tolist())
            if i % self.n_classes_per_task == (self.n_classes_per_task - 1):
                self.train_ds += [current_train]
                total_test.extend(current_test)
                self.cl_test_loaders.append(DataLoader(Subset(self.test_dataset, current_test), batch_size=self.args.batch_size, shuffle=False))
                self.full_test_loaders.append(DataLoader(Subset(self.test_dataset, deepcopy(total_test)), batch_size=self.args.batch_size, shuffle=False))
                current_train, current_test = [], []
        self.groups = defaultdict(list)
        for task_id in range(args.n_tasks):
            task_group = self.get_task_group(task_id, args.num_clients)
            for client in range(args.num_clients):
                data_i = task_group[client]
                self.groups[client].append(data_i)

    def get_task_group(self, task_id, num_users):
        train_indx = self.train_ds[task_id]
        targets = np.array(self.train_dataset.targets)[train_indx]
        groups, _ = create_lda_partitions(dataset=targets, num_partitions=num_users, concentration=self.args.alpha, accept_imbalanced=False)
        groups = [(np.array(train_indx)[groups[i][0]]).tolist() for i in range(num_users)]
        return groups

    def get_full_train(self, task_id):
        indexes = []
        for t in range(task_id + 1):
            indexes.extend(self.train_ds[t])
        return DataLoader(Subset(self.train_dataset, indexes), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    def get_full_test(self, t):
        return self.full_test_loaders[t]

    def get_cl_test(self, t):
        return self.cl_test_loaders[:t + 1]
