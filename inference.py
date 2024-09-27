import os
import argparse
import csv
import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataset import *
from model import *

def infer(args):

    # 设置结果的存储路径
    result_path = 'inference/'
    result_name = 'submission.csv'
    os.makedirs(result_path, exist_ok=True)

    # 测试集
    testset = LeavesDataset(mode='test')
    test_iter = DataLoader(testset, batch_size=1, num_workers=args.num_workers)

    # 指定训练用的设备
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # 实例化ResNet，加载训练好的模型
    model = get_custom_ResNet(input_channels=3 , num_classes=testset.num_classes)
    model.to(device)
    model.load_state_dict(torch.load(args.pth_file, map_location=device))

    # 验证模式
    model.eval()
    preds = []
    with torch.no_grad():
        for image in tqdm(test_iter):
            output = model(image.to(device))
            preds.append(output.argmax(dim=1).item())
    
    # 解码
    preds_str = [testset.num2class[i] for i in iter(preds)]

    # 保存并输出结果
    submission = pd.read_csv('./dataset/sample_submission.csv')
    assert len(submission['label']) == len(preds_str)
    submission['label'] = preds_str
    submission.to_csv(result_path + result_name, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_file', type=str, default='./wandb/run-20240928_013005-wtea8ary/files/model_2024_09_28_02_05_06_epoch99.pth')     # 改为训练完成的模型的存储地址
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=int, default=1)
    args = parser.parse_args()

    infer(args)