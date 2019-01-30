import traceback
import argparse
import sys
import os
import csv
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import vgg
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from hypergrad import SGDHD, AdamHD


class LogReg(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogReg, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = self.lin1(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self._input_dim = input_dim
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


def train(opt, log_func=None):
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.set_device(opt.device)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.enabled = True

    if opt.model == 'logreg':
        model = LogReg(28 * 28, 10)
    elif opt.model == 'mlp':
        model = MLP(28 * 28, 1000, 10)
    elif opt.model == 'vgg':
        model = vgg.vgg16_bn()
        if opt.parallel:
            model.features = torch.nn.DataParallel(model.features)
    else:
        raise Exception('Unknown model: {}'.format(opt.model))

    if opt.cuda:
        model = model.cuda()

    if opt.model == 'logreg' or opt.model == 'mlp':
        task = 'MNIST'
        train_loader = DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=opt.batchSize, shuffle=True)
        valid_loader = DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=opt.batchSize, shuffle=False)
    elif opt.model == 'vgg':
        task = 'CIFAR10'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=opt.batchSize, shuffle=True,
            num_workers=opt.workers, pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=opt.batchSize, shuffle=False,
            num_workers=opt.workers, pin_memory=True)
    else:
        raise Exception('Unknown model: {}'.format(opt.model))

    if opt.method == 'sgd':
        optimizer = SGD(model.parameters(), lr=opt.alpha_0, weight_decay=opt.weightDecay)
    elif opt.method == 'sgd_hd':
        optimizer = SGDHD(model.parameters(), lr=opt.alpha_0, weight_decay=opt.weightDecay, hypergrad_lr=opt.beta)
    elif opt.method == 'sgdn':
        optimizer = SGD(model.parameters(), lr=opt.alpha_0, weight_decay=opt.weightDecay, momentum=opt.mu, nesterov=True)
    elif opt.method == 'sgdn_hd':
        optimizer = SGDHD(model.parameters(), lr=opt.alpha_0, weight_decay=opt.weightDecay, momentum=opt.mu, nesterov=True, hypergrad_lr=opt.beta)
    elif opt.method == 'adam':
        optimizer = Adam(model.parameters(), lr=opt.alpha_0, weight_decay=opt.weightDecay)
    elif opt.method == 'adam_hd':
        optimizer = AdamHD(model.parameters(), lr=opt.alpha_0, weight_decay=opt.weightDecay, hypergrad_lr=opt.beta)
    else:
        raise Exception('Unknown method: {}'.format(opt.method))

    if not opt.silent:
        print('Task: {}, Model: {}, Method: {}'.format(task, opt.model, opt.method))

    model.eval()
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss = loss.data[0]
        break
    valid_loss = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        valid_loss += F.cross_entropy(output, target, size_average=False).data[0]
    valid_loss /= len(valid_loader.dataset)
    if log_func is not None:
        log_func(0, 0, 0, loss, loss, valid_loss, opt.alpha_0, opt.alpha_0, opt.beta)

    time_start = time.time()
    iteration = 1
    epoch = 1
    done = False
    while not done:
        model.train()
        loss_epoch = 0
        alpha_epoch = 0
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            loss = loss.data[0]
            loss_epoch += loss
            alpha = optimizer.param_groups[0]['lr']
            alpha_epoch += alpha
            iteration += 1
            if opt.iterations != 0:
                if iteration > opt.iterations:
                    print('Early stopping: iteration > {}'.format(opt.iterations))
                    done = True
                    break
            if opt.lossThreshold >= 0:
                if loss <= opt.lossThreshold:
                    print('Early stopping: loss <= {}'.format(opt.lossThreshold))
                    done = True
                    break

            if batch_id + 1 >= len(train_loader):
                loss_epoch /= len(train_loader)
                alpha_epoch /= len(train_loader)
                model.eval()
                valid_loss = 0
                for data, target in valid_loader:
                    data, target = Variable(data, volatile=True), Variable(target)
                    if opt.cuda:
                        data, target = data.cuda(), target.cuda()
                    output = model(data)
                    valid_loss += F.cross_entropy(output, target, size_average=False).data[0]
                valid_loss /= len(valid_loader.dataset)
                if log_func is not None:
                    log_func(epoch, iteration, time.time() - time_start, loss, loss_epoch, valid_loss, alpha, alpha_epoch, opt.beta)
            else:
                if log_func is not None:
                    log_func(epoch, iteration, time.time() - time_start, loss, float('nan'), float('nan'), alpha, float('nan'), opt.beta)

        epoch += 1
        if opt.epochs != 0:
            if epoch > opt.epochs:
                print('Early stopping: epoch > {}'.format(opt.epochs))
                done = True
    return loss, iteration


def main():
    try:
        parser = argparse.ArgumentParser(description='Hypergradient descent PyTorch tests', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--cuda', help='use CUDA', action='store_true')
        parser.add_argument('--device', help='selected CUDA device', default=0, type=int)
        parser.add_argument('--seed', help='random seed', default=1, type=int)
        parser.add_argument('--dir', help='directory to write the output files', default='results', type=str)
        parser.add_argument('--model', help='model (logreg, mlp, vgg)', default='logreg', type=str)
        parser.add_argument('--method', help='method (sgd, sgd_hd, sgdn, sgdn_hd, adam, adam_hd)', default='adam', type=str)
        parser.add_argument('--alpha_0', help='initial learning rate', default=0.001, type=float)
        parser.add_argument('--beta', help='learning learning rate', default=0.000001, type=float)
        parser.add_argument('--mu', help='momentum', default=0.9, type=float)
        parser.add_argument('--weightDecay', help='regularization', default=0.0001, type=float)
        parser.add_argument('--batchSize', help='minibatch size', default=128, type=int)
        parser.add_argument('--epochs', help='stop after this many epochs (0: disregard)', default=2, type=int)
        parser.add_argument('--iterations', help='stop after this many iterations (0: disregard)', default=0, type=int)
        parser.add_argument('--lossThreshold', help='stop after reaching this loss (0: disregard)', default=0, type=float)
        parser.add_argument('--silent', help='do not print output', action='store_true')
        parser.add_argument('--workers', help='number of data loading workers', default=4, type=int)
        parser.add_argument('--parallel', help='parallelize', action='store_true')
        parser.add_argument('--save', help='do not save output to file', action='store_true')
        opt = parser.parse_args()

        torch.manual_seed(opt.seed)
        if opt.cuda:
            torch.cuda.set_device(opt.device)
            torch.cuda.manual_seed(opt.seed)
            torch.backends.cudnn.enabled = True

        file_name = '{}/{}/{:+.0e}_{:+.0e}/{}.csv'.format(opt.dir, opt.model, opt.alpha_0, opt.beta, opt.method)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        if not opt.silent:
            print('Output file: {}'.format(file_name))
        # if os.path.isfile(file_name):
        #     print('File with previous results exists, skipping...')
        # else:
        if not opt.save:
            def log_func(epoch, iteration, time_spent, loss, loss_epoch, valid_loss, alpha, alpha_epoch, beta):
                if not opt.silent:
                    print('{} | {} | Epoch: {} | Iter: {} | Time: {:+.3e} | Loss: {:+.3e} | Valid. loss: {:+.3e} | Alpha: {:+.3e} | Beta: {:+.3e}'.format(opt.model, opt.method, epoch, iteration, time_spent, loss, valid_loss, alpha, beta))
            train(opt, log_func)
        else:
            with open(file_name, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Iteration', 'Time', 'Loss', 'LossEpoch', 'ValidLossEpoch', 'Alpha', 'AlphaEpoch', 'Beta'])
                def log_func(epoch, iteration, time_spent, loss, loss_epoch, valid_loss, alpha, alpha_epoch, beta):
                    writer.writerow([epoch, iteration, time_spent, loss, loss_epoch, valid_loss, alpha, alpha_epoch, beta])
                    if not opt.silent:
                        print('{} | {} | Epoch: {} | Iter: {} | Time: {:+.3e} | Loss: {:+.3e} | Valid. loss: {:+.3e} | Alpha: {:+.3e} | Beta: {:+.3e}'.format(opt.model, opt.method, epoch, iteration, time_spent, loss, valid_loss, alpha, beta))
                train(opt, log_func)

    except KeyboardInterrupt:
        print('Stopped')
    except Exception:
        traceback.print_exc(file=sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
