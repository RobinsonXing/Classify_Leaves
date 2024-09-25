import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Trainset(Dataset):
    def __init__(self):
        # 读取数据集
        self.data = pd.read_csv('./dataset/train.csv')
        # 预处理和增强
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        image = self.transform(Image.open(image_path).convert('RGB'))
        return image, label


