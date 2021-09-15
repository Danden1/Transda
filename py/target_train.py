import random
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import numpy as np

from model import Transda
from dataset import MyDataset
from dataset import split_data
from loss import CrossEntropy


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


def self_labeling(model, datas, batch_size, i, device):
    start_chk = True

    with torch.no_grad():
        for idx in range(i):
            try:
                data = datas[idx * batch_size: idx * batch_size + batch_size].float().to(device)
            except:
                data = datas[idx * batch_size:].float().to(device)

            feature, target = model(data)
            target = F.softmax(target, dim=1)

            if start_chk:
                start_chk = False
                features = feature.float()
                targets = target.float()

            else:
                features = torch.cat((features, feature.float()), dim=0)
                targets = torch.cat((targets, target.float()), dim=0)

    centroid = targets.transpose(0, 1).matmul(features) / (1e-8 + targets.sum(axis=0)[:, None])
    similarity = torch.tensor([]).to(device)

    for i in range(features.size(0)):
        similarity = torch.cat((similarity, F.cosine_similarity(features[i].unsqueeze(0), centroid).unsqueeze(0)),
                               dim=0)

    self_label = torch.zeros((features.size(0), targets.size(1))).to(device)
    # not distance, so argmax

    tmp_label = torch.argmax(similarity, dim=1)
    for i in range(self_label.size(0)):
        self_label[i][tmp_label[i]] = 1

    centroid = self_label.transpose(0, 1).matmul(features) / (1e-8 + self_label.sum(axis=0)[:, None])

    similarity = torch.tensor([]).to(device)
    for i in range(features.size(0)):
        similarity = torch.cat((similarity, F.cosine_similarity(features[i].unsqueeze(0), centroid).unsqueeze(0)),
                               dim=0)

    self_label = torch.argmax(similarity, dim=1).unsqueeze(0)
    self_label = self_label.view(-1, 1)

    return self_label.long().to(device), 1. - similarity


def target_train(train_datas, test_datas, device, target_domain, args):
    for j in range(len(target_domain)):
        domain = target_domain[j]
        train_data = train_datas[j]
        test_data = test_datas[j]

        student_model = Transda(num_classes=31)
        student_model = student_model.to(device)
        student_model.load_state_dict(torch.load(args.chkp_path, map_location=device))

        teacher_model = Transda(num_classes=31)
        teacher_model = teacher_model.to(device)
        teacher_model.load_state_dict(torch.load(args.chkp_path, map_location=device))

        for i in teacher_model.parameters():
            i.required_grad = False

        for i in student_model.cls.parameters():
            i.required_grad = False

        celoss = CrossEntropy()

        optimizer = optim.SGD([{'params': student_model.feature.parameters(), 'lr': args.lr * 0.1},
                               {'params': student_model.fn.parameters()}], lr=args.lr)

        optimizer = op_copy(optimizer)
        train_dataset = MyDataset(train_data, transforms=train_transforms)
        test_dataset = MyDataset(test_data, transforms=test_transforms)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

        train_accs = []
        train_losses = []

        test_accs = []
        for epoch in range(1, args.epoch + 1):
            lr_scheduler(optimizer, iter_num=epoch, max_iter=args.epoch)
            # self_labeling
            datas = torch.tensor([]).float().to(device)
            targets = torch.tensor([]).long().to(device)
            i = 0
            for idx, (data, target) in enumerate(train_dataloader):
                data = data.float().to(device)
                target = target.long().to(device)

                datas = torch.cat((datas, data), dim=0)
                targets = torch.cat((targets, target), dim=0)
                i = idx

            teacher_model.eval()
            student_model.eval()
            self_labels, distances = self_labeling(teacher_model, datas, args.batch_size, i, device)
            student_model.train()

            train_loss = 0.
            train_acc = 0.

            test_acc = 0.

            for idx in range(i):
                try:
                    data = datas[idx * args.batch_size: idx * args.batch_size + args.batch_size].float().to(device)
                    target = targets[idx * args.batch_size: idx * args.batch_size + args.batch_size].long().to(device)
                except:
                    data = datas[idx * args.batch_size:].float().to(device)
                    target = targets[idx * args.batch_size:].long().to(device)

                _, pred = student_model(data)
                softmax_output = F.softmax(pred, dim=-1)

                im_loss = celoss(softmax_output, softmax_output)
                p = torch.mean(softmax_output, dim=0)
                p = (p * torch.log(p + 1e-5)).sum()

                im_loss += p

                self_label = self_labels[idx * data.size(0): idx * data.size(0) + data.size(0)]
                self_label = torch.zeros((data.size(0), pred.size(1))).to(device)
                for i in range(data.size(0)):
                    self_label[i][self_labels[idx * data.size(0) + i]] = 1
                sl_loss = celoss(self_label, softmax_output)

                distance = distances[idx * data.size(0):idx * data.size(0) + data.size(0)]
                soft_label = torch.exp(distance / 0.1) / (torch.sum(torch.exp(distance / 0.1), dim=1)[:, None] + 1e-8)
                kd_loss = celoss(soft_label, softmax_output)

                tgt_loss = im_loss + 0.3 * sl_loss + kd_loss

                optimizer.zero_grad()
                tgt_loss.backward()
                optimizer.step()

                train_loss += tgt_loss
                train_acc += (pred.argmax(dim=1) == target).float().mean()

            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # ema

            with torch.no_grad():
                m = 0.001  # momentum parameter
                for param_q, param_k in zip(student_model.feature.parameters(), teacher_model.feature.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for i in teacher_model.parameters():
                i.required_grad = False

            student_model.eval()
            for data, target in test_dataloader:
                data = data.float().to(device)
                target = target.long().to(device)
                with torch.no_grad():
                    _, pred = student_model(data)
                    test_acc += (pred.argmax(dim=1) == target).float().mean()

            test_acc /= len(test_dataloader)
            test_accs.append(test_acc)

            print('domain:{0} -> {1} : epoch {2}, train_acc : {3}, test_acc : {4}' .format(domain_names[args.s], domain, epoch, train_acc, test_acc))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0')
    parser.add_argument('--s', type=int, default=0, help='source domain')
    parser.add_argument('--epoch', type=int, default=15, help='epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--data_path', type=str, default='../../../shared/domain_adaptation/office')
    parser.add_argument('--chkp_path', type=str, default='./chkp/0.pt')
    args = parser.parse_args()

    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)

    DATA_PATH = [args.data_path+'/amazon/images/', args.data_path+'/dslr/images/', args.data_path+'/webcam/images/']
    class_list = os.listdir(DATA_PATH[args.s])

    device = 'cuda:' + args.gpu_id

    train_datas, test_datas= [],[]
    target_domain = domain_names[:]
    del target_domain[args.s]

    for i in range(len(domain_names)):
        if domain_names[i] in target_domain:
            train_data, test_data = split_data(DATA_PATH[i], class_list)
            train_datas.append(train_data)
            test_datas.append(test_data)

    target_train(train_datas, test_datas, device, target_domain, args)
