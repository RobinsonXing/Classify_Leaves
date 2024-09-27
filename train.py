import random
import argparse
import datetime
import os
import logging
import tqdm
import wandb

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import *
from model import *

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    # 设置wandb
    wandb.init(project='Kaggle: Classify Leaves', config=args)
    wandb.run.name = 'ResNet_custom'
    wandb.run.save()

    # 加载训练集和测试集
    trainset = LeavesDataset(mode='train')
    validset = LeavesDataset(mode='valid')
    train_iter = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    valid_iter = DataLoader(validset, batch_size=args.batch_size, shuffle=False)

    # 通过cuda:<device_id>指定训练用的GPU
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # 实例化ResNet，加载至GPU上
    model = get_custom_ResNet(trainset.num_classes).to(device)

    # 指定Adam优化器、交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch_id in range(args.num_epochs):

        model.train()
        for batch_id, batch in tqdm(enumerate(train_iter)):

            # 按批次将数据加载至GPU中
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 后向传播
            loss.backward()
            optimizer.step()
            
            # 计算
            step = epoch_id * len(train_iter) + batch_id
            train_loss = loss.cpu().item()
            train_acc = (outputs.argmax(dim=1) == labels).sum().cpu().item()

        

if __name__ == '__main__':

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_dacay', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)