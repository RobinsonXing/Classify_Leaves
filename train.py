import random
import argparse
import datetime
import os
import logging
import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import *
from model import *

def set_random_seed(seed=0):
    random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):

    # 设置日志名和路径名
    name = f'batchsize{args.batch_size}_epochs{args.epochs}_cuda{args.cuda}_lr{args.lr}_seed{args.seed}'
    current_time = datetime.datetime.now()
    log_path = os.path.join('log', 'task1', name)
    checkpoint_path = os.path.join('checkpoint', 'task1', name)

    # 创建保存日志和模型的路径
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(log_path, f'{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.txt'),
                        filemode='w')

    # 加载训练集
    dataset_train = Trainset()
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    # 通过cuda:<device_id>指定训练用的GPU
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # 实例化ResNet，加载至GPU上
    model = ResNetClassifier(dataset_train.num_classes).to(device)

    # 指定Adam优化器、交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in tqdm(range(args.num_epochs)):

        # 训练模式
        model.train()

        running_loss = 0.0
        for images, labels in dataloader_train:

            # 按批次将数据加载至GPU中
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 后向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 日志记录
        logging.info(f'epoch: {epoch}, loss: {running_loss/len(dataloader_train):.4f}')

        # 保存模型
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'{current_time.strftime("%Y_%m_%d_%H_%M_%S")}.pth'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=176)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    set_random_seed(args.seed)
    train(args)