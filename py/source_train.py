import random
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import os
import numpy as np

from model import Transda
from dataset import MyDataset
from dataset import split_data
from loss import SmoothingCrossEntropy

train_transforms = transforms.Compose([
      transforms.Resize((256,256)),
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_transforms = transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor(),
      #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

domain_names = ['amazon', 'dslr', 'webcam']

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True

    return optimizer



def pretrain(train_data, test_data, device, args):
    pretrain_model = Transda(num_classes=31)
    pretrain_model.to(device)


    train_dataset = MyDataset(train_data, train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = MyDataset(test_data, test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = optim.SGD([
        {'params': pretrain_model.feature.parameters(), 'lr': args.lr * 0.1},
        {'params': pretrain_model.cls.parameters()},
        {'params': pretrain_model.fn.parameters()}], lr=args.lr)

    optimizer = op_copy(optimizer)
    CEloss = SmoothingCrossEntropy(num_classes=31)

    train_losses = []
    train_accs = []

    test_losses = []
    test_accs = []

    for epoch in range(1, args.epoch + 1):

        train_loss = 0.
        train_acc = 0.

        test_loss = 0.
        test_acc = 0.

        lr_scheduler(optimizer, iter_num=epoch, max_iter=args.epoch)
        pretrain_model.train()
        for data, target in train_loader:
            data = data.float().to(device)
            target = target.long().to(device)

            optimizer.zero_grad()
            _, pred = pretrain_model(data)
            loss = CEloss(pred, target)
            loss.backward()

            train_loss += loss
            train_acc += (pred.argmax(dim=1) == target).float().mean()

            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        pretrain_model.eval()

        for data, target in test_loader:
            data = data.float().to(device)
            target = target.long().to(device)

            with torch.no_grad():
                _, pred = pretrain_model(data)
                loss = CEloss(pred, target)
                test_loss += loss
                test_acc += (pred.argmax(dim=1) == target).float().mean()

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print('domain : {0}, : epoch {1}, train_acc : {2}, test_acc : {3}'.format(domain_names[args.s], epoch, train_acc, test_acc))


    if args.chkp_path is not None:
        torch.save(pretrain_model.state_dict(), args.chkp_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0')
    parser.add_argument('--s', type=int, default=0, help='source domain')
    parser.add_argument('--t', type=int, default=1, help='target domain')
    parser.add_argument('--epoch', type=int, default=20, help='epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--data_path', type=str, default='../../../../shared/domain_adaptation/office')
    parser.add_argument('--chkp_path', type=str, default='./chkp/0.pt')
    parser.add_argument('--split', type=float, default=0.9)
    args = parser.parse_args()

    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)

    DATA_PATH = [args.data_path+'/amazon/images/', args.data_path+'/dslr/images/', args.data_path+'/webcam/images/']
    class_list = os.listdir(DATA_PATH[args.s])

    device = 'cuda:' + args.gpu_id
    train_data, test_data = split_data(DATA_PATH[args.s], class_list, args.split)

    pretrain(train_data, test_data, device, args)


