import numpy as np
import pandas as pd
from PIL import Image
from typing import Literal

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LeavesDataset(Dataset):
    """读取数据集"""
    def __init__(self, csv_path,
                 valid_ratio=0.2, resize_height=224, resize_width=224,
                 mode: Literal['train', 'valid', 'test']='train'):

        # 读取数据集
        self.data = pd.read_csv(csv_path, header=None)

        # 计算验证集长度
        data_len = len(self.data.index)
        train_len = int(data_len * (1 - valid_ratio))\

        # 按mode加载数据
        self.mode = mode
        if self.mode == 'train':
            self.image_arr = np.asarray(self.data.iloc[:train_len, 0])
            self.label_arr = np.asarray(self.data.iloc[:train_len, 1])
        elif self.mode == 'valid':
            self.image_arr = np.asarray(self.data.iloc[train_len:, 0])
            self.label_arr = np.asarray(self.data.iloc[train_len:, 1])
        elif self.mode == 'test':
            self.image_arr = np.asarray(self.data.iloc[:, 0])
            self.label_arr = None
        
        # 测试打印
        print(f'Finished loading the {mode}set with {len(self.image_arr)} datarow.')

        # 数据增强和预处理
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((resize_height, resize_width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resize_height, resize_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # 标签编码
        def load_calss2num():
            data = pd.read_csv('./dataset/train.csv')
            unique_labels = sorted(list(data['label'].unique()))
            return dict(zip(unique_labels, range(len(unique_labels))))
        self.calss2num = load_calss2num()
        self.num2class = {v : k for k, v in self.calss2num.items()}

    def __len__(self):
        return len(self.image_arr)
    
    def __getitem__(self, index):
        image_path = self.image_arr[index]
        image = self.transform(Image.open('./dataset/' + image_path).convert('RGB'))
        if self.mode == 'test':
            return image
        else:
            label_num = self.calss2num[self.label_arr[index]]
            return image, label_num