import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader, Subset

from constant import *


class AVG:
    def __init__(self, batch_size, epochs, train_dataset, groups, dataset_name):
        self.criterion_fn = F.cross_entropy
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.groups = groups
        self.current_t = -1
        self.local_epoch = epochs
        self.dataset_name = dataset_name

    def set_dataloader(self, samples):
        if self.dataset_name in [CIFAR100, tinyImageNet]:
            self.train_loader = DataLoader(Subset(self.train_dataset, samples), batch_size=self.batch_size, shuffle=True)
        if self.dataset_name == SuperImageNet:
            self.train_loader = self.train_dataset.get_dl(samples, train=True)

    def set_next_t(self):
        self.current_t += 1
        samples = self.groups[self.current_t]
        self.set_dataloader(samples)

    def train(self, model, lr, teacher, generator_server, glob_iter_):
        model.to('cuda')
        model.train()
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        for epoch in range(self.local_epoch):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to('cuda'), y.to('cuda')
                logits = model(x)
                loss = self.criterion_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.to('cpu')


class PROX(AVG):
    def __init__(self, batch_size, epochs, train_dataset, groups, dataset_name):
        super(PROX, self).__init__(batch_size, epochs, train_dataset, groups, dataset_name)
        self.mu = 0.01

    def train(self, model, lr, teacher, generator_server, glob_iter_):
        model.to('cuda')
        model.train()
        global_model = deepcopy(model)
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        for epoch in range(self.local_epoch):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to('cuda'), y.to('cuda')
                logits = model(x)
                opt.zero_grad()
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = self.criterion_fn(logits, y) + (self.mu / 2) * proximal_term
                loss.backward()
                opt.step()
        model.to('cpu')


class ORACLE(AVG):
    def __init__(self, batch_size, epochs, train_dataset, groups, dataset_name):
        super(ORACLE, self).__init__(batch_size, epochs, train_dataset, groups, dataset_name)

    def set_next_t(self):
        self.current_t += 1
        current_group = []
        for task in range(self.current_t + 1):
            current_group.extend(self.groups[task])
        self.set_dataloader(current_group)

    def train(self, model, lr, teacher, generator_server, glob_iter_):
        model.to('cuda')
        model.train()
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        for epoch in range(self.local_epoch):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to('cuda'), y.to('cuda')
                logits = model(x)
                loss = self.criterion_fn(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.to('cpu')
