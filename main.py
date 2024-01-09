import os
import numpy as np
from copy import deepcopy

import models
from constant import *
from clients.MFCL import MFCL
from models.ResNet import ResNet18
from models.myNetwork import network
from data_prep.data import CL_dataset
from clients.simple import AVG, PROX, ORACLE
from data_prep.super_imagenet import SuperImageNet
from utiles import setup_seed, fedavg_aggregation, evaluate_accuracy_forgetting, evaluate_accuracy, train_gen, start


args = start()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuID
setup_seed(args.seed)

if args.dataset == CIFAR100:
    dataset = CL_dataset(args)
    feature_extractor = ResNet18(args.num_classes, cifar=True)
    ds = dataset.train_dataset
elif args.dataset == tinyImageNet:
    dataset = CL_dataset(args)
    feature_extractor = ResNet18(args.num_classes, cifar=False)
    ds = dataset.train_dataset
    args.generator_model = 'TINYIMNET_GEN'
elif args.dataset == SuperImageNet:
    from models.imagenet_resnet import resnet18
    dataset = SuperImageNet(args.path, version=args.version, num_tasks=args.n_tasks, num_clients=args.num_clients, batch_size=args.batch_size)
    args.num_classes = dataset.num_classes
    feature_extractor = resnet18(args.num_classes)
    args.generator_model = 'IMNET_GEN'
    args.img_size = dataset.img_size
    ds = dataset

global_model = network(dataset.n_classes_per_task, feature_extractor)
teacher, generator = None, None
gamma = np.log(args.lr_end / args.lr)
task_size = dataset.n_classes_per_task
counter, classes_learned = 0, task_size
num_participants = int(args.frac * args.num_clients)
clients, max_accuracy = [], []
if args.method == MFCL:
    generator = models.__dict__['generator'].__dict__[args.generator_model](zdim=args.z_dim, convdim=args.conv_dim)

for i in range(args.num_clients):
    group = dataset.groups[i]
    if args.method == FedAVG:
        client = AVG(args.batch_size, args.epochs, ds, group, args.dataset)
    elif args.method == FedProx:
        client = PROX(args.batch_size, args.epochs, ds, group, args.dataset)
    elif args.method == ORACLE:
        client = ORACLE(args.batch_size, args.epochs, ds, group, args.dataset)
    elif args.method == MFCL:
        client = MFCL(args.batch_size, args.epochs, ds, group, args.client_type, args.w_kd, args.w_ft, args.syn_size, args.dataset)
    clients.append(client)

for t in range(args.n_tasks):
    test_loader = dataset.get_full_test(t)
    [client.set_next_t() for client in clients]
    for round in range(args.global_round):
        weights = []
        lr = args.lr * np.exp(round / args.global_round * gamma)
        selected_clients = [clients[idx] for idx in np.random.choice(args.num_clients, num_participants, replace=False)]
        for user in selected_clients:
            model = deepcopy(global_model)
            user.train(model, lr, teacher, generator, counter)
            weights.append(model.state_dict())
        global_model.load_state_dict(fedavg_aggregation(weights))
        if (round + 1) % args.eval_int == 0:
            correct, total = evaluate_accuracy(global_model, test_loader, args.method)
            print(f'round {counter}, accuracy: {100 * correct / total}')
        counter += 1
    if t == 0:
        max_accuracy.append(correct / total)
    if t > 0:
        correct, total, accuracies = evaluate_accuracy_forgetting(global_model, dataset.get_cl_test(t), args.method)
        print(f"total_accuracy_{t}: {accuracies}")
        max_accuracy.append(accuracies[-1])
    if t != args.n_tasks - 1:
        if args.method == MFCL:
            original_global = deepcopy(global_model)
            teacher = train_gen(deepcopy(global_model), classes_learned, generator, args)
            for client in clients:
                client.last_valid_dim = classes_learned
                client.valid_dim = classes_learned + task_size
            global_model = original_global
        classes_learned += task_size

print('forgetting:', sum([max_accuracy[i] - accuracies[i] for i in range(args.n_tasks)]) / args.n_tasks)
