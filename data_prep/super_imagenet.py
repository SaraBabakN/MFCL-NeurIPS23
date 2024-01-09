import os
import random
from pathlib import Path
from collections import defaultdict
import yaml
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import PIL
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, transform=None):
        self.data_frame = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data_frame.iloc[idx]["file_name"]
        img_path = img_name
        image = PIL.Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.data_frame.iloc[idx].get("label", -1)


def get_dataloader(batch_size, test_list, trans, shuffle):
    dataset = ImageDataset(pd.DataFrame(test_list), transform=trans)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=shuffle, num_workers=8, pin_memory=True)
    return loader


class SuperImageNet():
    idx_to_wnid = 'super-imagenet-class-mapping.yaml'
    version_classlist = 'super-imagenet-versions-classlist.yaml'
    train_folder_name = 'train'
    val_folder_name = 'val'
    meta_folder = os.path.dirname(Path(__file__).absolute())
    valid_image_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    num_samples_per_version_dict = {'train': {'S': 2500, 'M': 5000, 'L': 7500}, 'val': {'S': 100, 'M': 200, 'L': 300}}
    num_classes_per_version_dict = {'S': 100, 'M': 75, 'L': 50}

    def __init__(self, root, version='M', num_tasks=1, num_clients=1, batch_size=16):
        """
        Implementation of SuperImageNet-S/M/L dataset. Can be customized to split classes into disjoint sets for continual learning tasks.
        For class-incremental setup, specify total number of tasks, and the current task to generate data for.
        For federated setup specify total number of clients, and the client index to generate data for.
        Samples will be uniformly drawn for each client over available classes (same number of samples for each class).
        Arguments:
        - root: root directory of ImageNet data, should include 'train' and 'val' folders for training and validation splits
        - version:
            'S': 100 classes, 2500 samples/class
            'M': 75 classes, 5000 samples/class
            'L': 50 classes, 7500 samples/class
        - num_tasks: total number of tasks
        - num_clients: total number of clients in federated setting
        """
        self.version = version
        self.num_tasks = num_tasks
        self.num_clients = num_clients
        dset_mean = (0.485, 0.456, 0.406)
        dset_std = (0.229, 0.224, 0.225)
        self.img_size = 224
        self.transform = transforms.Compose([transforms.RandomResizedCrop(self.img_size),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(dset_mean, dset_std)])

        self.test_transform = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(self.img_size),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(dset_mean, dset_std)])

        self.batch_size = batch_size
        self.class_maping = {}

        # Get dataset info
        self.num_classes = self.num_classes_per_version_dict[self.version]
        assert self.num_classes % num_tasks == 0
        self.n_classes_per_task = self.num_classes // self.num_tasks

        # Load class index - WordNetID mapping
        idx_to_wnid_file = Path(self.meta_folder) / Path(self.idx_to_wnid)
        with Path(idx_to_wnid_file).open('r') as f:
            idx_to_wnid_dict = yaml.safe_load(f)

        # Load class indices based on version (-S, -M, -L)
        version_classlist_file = Path(self.meta_folder) / Path(self.version_classlist)
        with Path(version_classlist_file).open('r') as f:
            version_classlist_dict = yaml.safe_load(f)
            version_classlist = version_classlist_dict[self.version]

        # Get data folder
        self.data = {}
        self.idx_to_classname = {}
        folders = [self.train_folder_name, self.val_folder_name]
        for folder_name in folders:
            self.read_data(folder_name, root, idx_to_wnid_dict, version_classlist)
        self.class_idx_to_keep = list(self.data[self.train_folder_name].keys())
        random.shuffle(self.class_idx_to_keep)
        for i in range(len(self.class_idx_to_keep)):
            self.class_maping[self.class_idx_to_keep[i]] = i
        self.groups = defaultdict(list)
        self.task_data = defaultdict(list)
        self.test_data = []
        for t in range(self.num_tasks):
            self.set_new_task(t)

    def read_data(self, folder_name, root, idx_to_wnid_dict, version_classlist):
        data_dir = Path(root) / Path(folder_name)
        self.data[folder_name] = defaultdict(list)
        num_samples_per_class = self.num_samples_per_version_dict[folder_name][self.version]
        assert num_samples_per_class % self.num_clients == 0
        for idx, v in idx_to_wnid_dict.items():
            if idx in version_classlist:
                class_data = []
                for q in v[1:]:
                    for k in q:
                        image_dir = data_dir / Path(k)
                        images = list(image_dir.iterdir())
                        images = [im for im in images if im.suffix in self.valid_image_extensions]
                        class_data.extend(images)
                random.shuffle(class_data)
                self.data[folder_name][idx] = class_data[:num_samples_per_class]

    def get_full_test(self, task_id):
        data = []
        for t in range(task_id + 1):
            data.extend(self.test_data[t])
        return self.get_dl(data, train=False)

    def get_cl_test(self, task_id):
        dls = []
        for t in range(task_id + 1):
            dls.append(self.get_dl(self.test_data[t], train=False))
        return dls

    def get_full_train(self, task_id):
        data = []
        for task in range(task_id + 1):
            data.extend(self.task_data[task])
        return self.get_dl(data)

    def set_new_task(self, task_idx):
        assert 0 <= task_idx < self.num_tasks
        self.task_class_idx_train = sorted(self.class_idx_to_keep[self.n_classes_per_task * task_idx: self.n_classes_per_task * (task_idx + 1)])
        print(f'task {task_idx}, classes: {self.task_class_idx_train}')
        sample_per_client = self.num_samples_per_version_dict[self.train_folder_name][self.version] // self.num_clients
        current_client_data = defaultdict(list)
        sample_idx_to_keep = []
        for current_class in self.task_class_idx_train:
            label = self.class_maping[current_class]
            samples_for_class = self.data[self.train_folder_name][current_class]
            random.shuffle(samples_for_class)
            samples_for_class = [{'file_name': str(i), 'label': label} for i in samples_for_class]
            self.task_data[task_idx].extend(samples_for_class)
            for idx in range(self.num_clients):
                current_client_data[idx].extend(samples_for_class[sample_per_client * idx: sample_per_client * (idx + 1)])
            samples_for_class = self.data[self.val_folder_name][current_class]
            samples_for_class = [{'file_name': str(i), 'label': label} for i in samples_for_class]
            sample_idx_to_keep.extend(samples_for_class)
        self.test_data.append(sample_idx_to_keep)
        for idx in range(self.num_clients):
            self.groups[idx].append(current_client_data[idx])

    def get_dl(self, samples, train=True):
        trans = self.transform if train else self.test_transform
        bs = self.batch_size if train else 128
        return get_dataloader(bs, samples, trans, shuffle=train)
