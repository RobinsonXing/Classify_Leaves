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

    # 加载训练集和验证集
    trainset = LeavesDataset(mode='train')
    validset = LeavesDataset(mode='valid')
    train_iter = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valid_iter = DataLoader(validset, batch_size=args.batch_size, shuffle=False)

    # 指定训练用的设备
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # 实例化ResNet，加载至GPU上
    model = get_custom_ResNet(trainset.num_classes)
    model = model.to(device)

    # 优化器、学习率调度器、损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch_id in range(args.num_epochs):

        model.train()
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        print(f'Starting training for epoch {epoch_id + 1}')

        for batch in tqdm(train_iter):

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
            
            # 累积损失和准确率
            running_loss += loss.item() * images.size(0)
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_predictions += labels.size(0)
        
        # 计算训练集平均损失和准确率
        train_loss = running_loss / len(trainset)
        train_accuracy = correct_predictions / total_predictions
        print(f'Epoch {epoch_id + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

        # 验证模式
        model.eval()
        valid_loss, valid_correct, valid_total = 0.0, 0, 0
        print(f'Starting validation for epoch {epoch_id}')

        # 置零梯度，不更新参数
        with torch.no_grad():
            for batch in valid_iter:

                # 按批次将数据加载至GPU中
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                # 推断
                outputs = model(images)

                # 计算损失和精度
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)
                valid_correct += (outputs.argmax(dim=1) == labels).sum().item()
                valid_total += labels.size(0)

        # 计算指标
        valid_loss /= len(validset)
        valid_accuracy = valid_correct / valid_total
        print(f'Epoch {epoch_id + 1}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

        # 记录到wandb中
        wandb.log({
            'Train Loss': train_loss,
            'Valid Loss': valid_loss
        })
        wandb.log({
            'Train Accuracy': train_accuracy,
            'Valid Accuracy': valid_accuracy
        })

        # 保存模型到WandB
        torch.save(model.state_dict(), f'model_epoch_{epoch_id + 1}.pth')
        wandb.save(f'model_epoch_{epoch_id + 1}.pth')

        # 调度器更新
        scheduler.step()

        

if __name__ == '__main__':

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_dacay', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)