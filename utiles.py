import torch
import random
import argparse
import numpy as np
from copy import deepcopy

from constant import *
from clients.helper import Teacher


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    # torch.cuda.empty_cache()


def fedavg_aggregation(weights):
    w_avg = deepcopy(weights[0])
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights))
    return w_avg


def evaluate_accuracy(model, test_loader, method=None):
    model.to('cuda')
    model.eval()
    correct, total = 0, 0
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to('cuda'), y.to('cuda')
        with torch.no_grad():
            outputs = model(x)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == y.cpu()).sum()
        total += len(y)
    model.to('cpu')
    return correct, total


def evaluate_accuracy_forgetting(model, test_loaders, method=None):
    c, t = 0, 0
    accuracies = []
    for task_id, test_loader in enumerate(test_loaders):
        ci, ti = evaluate_accuracy(model, test_loader, method)
        accuracies.append(100 * ci / ti)
        c += ci
        t += ti
    return c, t, accuracies


def train_gen(model, valid_out_dim, generator, args):
    dataset_size = (-1, 3, args.img_size, args.img_size)
    model.to('cuda')
    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=args.generator_lr)
    teacher = Teacher(solver=model, generator=generator, gen_opt=generator_optimizer,
                      img_shape=dataset_size, iters=args.pi, deep_inv_params=[1e-3, args.w_bn, args.w_noise, 1e3, 1],
                      class_idx=np.arange(valid_out_dim), train=True, args=args)
    teacher.sample(args.server_ss, return_scores=False)
    return teacher, deepcopy(model.fc)


def combine_data(data):
    x, y = [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    x, y = torch.cat(x), torch.cat(y)
    return x, y


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuID', type=str, default='0', help="GPU ID")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--method', type=str, default=MFCL, help="name of method", choices=[FedAVG, FedProx, MFCL])
    parser.add_argument('--dataset', type=str, default=CIFAR100, help="name of dataset")
    parser.add_argument('--num_clients', type=int, default=50, help='#clients')
    parser.add_argument('--epochs', type=int, default=10, help='Local Epoch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Local Learning Rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Local Bachsize')
    parser.add_argument('--eval_int', type=int, default=10, help='Evaluation intervals')
    parser.add_argument('--global_round', type=int, default=100, help='#global rounds per task')
    parser.add_argument('--frac', type=float, default=0.1, help='#selected clients in each round')
    parser.add_argument('--alpha', type=float, default=1, help='LDA parameter for data distribution')
    parser.add_argument('--n_tasks', type=int, default=10, help='#tasks')
    parser.add_argument('--syn_size', type=int, default=128, help='size of mini-batch')
    parser.add_argument('--server_ss', type=int, default=128, help='batch size for genrative training')
    parser.add_argument('--pi', type=int, default=100, help='local epochs of each global round')
    parser.add_argument('--generator_lr', type=float, default=0.001)
    parser.add_argument('--z_dim', type=int, default=1000)
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--ie_loss', type=int, default=1)
    parser.add_argument('--act_loss', type=int, default=0)
    parser.add_argument('--bn_loss', type=int, default=1)
    parser.add_argument('--noise', type=int, default=1)
    parser.add_argument('--w_ie', type=float, default=1.)
    parser.add_argument('--w_kd', type=float, default=1e-1)
    parser.add_argument('--w_ft', type=float, default=1)
    parser.add_argument('--w_act', type=float, default=0.1)
    parser.add_argument('--w_noise', type=float, default=1e-3)
    parser.add_argument('--w_bn', type=float, default=5e1)
    parser.add_argument('--generator_model', type=str, default='CIFAR_GEN', help='name of the generative model')
    parser.add_argument('--path', type=str, help='path to dataset')
    parser.add_argument('--version', type=str, default='L')
    args = parser.parse_args()
    args.lr_end = 0.01
    return args
