import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Trainset(Dataset):
    def __init__(self):
        # 读取数据集
        self.data = pd.read_csv('./dataset/train.csv')
        # 

