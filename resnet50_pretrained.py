import sys
from glob import glob
import cv2
# import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from tqdm import tqdm
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import *
from torch import nn
import math
from torch.optim.lr_scheduler import StepLR

class Mydatasets(Dataset):
    def __init__(self, datasets, transform):
        self.transformer = transform
        self.dirname = datasets[0]
        self.label = datasets[1]
        self.files = glob(f"{self.dirname}/*")

    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        image = self.transformer(image)
        return image, self.label



def main():
    nonTC = "nonTC"
    TC = "TC"

    num_epochs = 50

    datasets_path_0 = {
        f"./train_1/train/{nonTC}":0,
        f"./train_2/train/{nonTC}":0,
        f"./train_3/train/{nonTC}":0,
        f"./train_4/train/{nonTC}":0,
        f"./train_5/train/{nonTC}":0,
        f"./train_6/train/{nonTC}":0,
        f"./train_7/train/{nonTC}":0,
        f"./train_8/train/{nonTC}":0,
        f"./train_9/train/{nonTC}":0,
        f"./train_10/train/{nonTC}":0,
        f"./train_11/train/{nonTC}":0,
        f"./train_12/train/{nonTC}":0,
        f"./train_13/train/{nonTC}":0,
        f"./train_14/train/{nonTC}":0,
        # f"./train_15/train/{nonTC}":0,
        # f"./train_16/train/{nonTC}":0,
    }
    datasets_path_1 = {
        f"./train_1/train/{TC}":1,
        f"./train_2/train/{TC}":1,
        f"./train_3/train/{TC}":1,
        f"./train_4/train/{TC}":1,
        f"./train_5/train/{TC}":1,
        f"./train_6/train/{TC}":1,
        f"./train_7/train/{TC}":1,
        f"./train_8/train/{TC}":1,
        f"./train_9/train/{TC}":1,
        f"./train_10/train/{TC}":1,
        f"./train_11/train/{TC}":1,
        f"./train_12/train/{TC}":1,
        f"./train_13/train/{TC}":1,
        f"./train_14/train/{TC}":1,
        # f"./train_15/train/{TC}":1,
        # f"./train_16/train/{TC}":1,
    }
    mix_path = {
        f"./mix//{TC}":1,
        f"./mix/{nonTC}":0,
    }

    # transform作成
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),  # 224x224にリサイズ
        transforms.ToTensor(),          # テンソルに変換
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # クラスごとのサンプル数
    class_counts = []
    total_samples = np.sum(class_counts)
    
    # クラスごとの重み
    class_weights = total_samples / (len(class_counts) * class_counts)

    # クラス1の総枚数と１データセット当たりの平均枚数算出
    # クラス1のデータセット作成
    data = []
    for datasets in datasets_path_1.items():
        set = Mydatasets(datasets, transform)
        data.append(set)
    datas_1 = ConcatDataset(data)
    
    class1_num = len(datas_1)
    class1_num_per = math.floor(class1_num/len(datasets_path_1))
    print(f"class1_num:{class1_num}, class1_num_per:{class1_num_per}")

    # クラス0の総枚数と１データセット当たりの平均枚数算出
    # クラス0のデータセット作成
    data = []
    class0_num = 0
    for datasets in datasets_path_0.items():
        set = Mydatasets(datasets, transform)
        class0_num += len(set)
        indices = np.random.choice(len(set), class1_num_per, replace=False)
        set = Subset(set, indices)
        data.append(set)
    datas_0 = ConcatDataset(data)

    print(f"datas_0:{len(datas_0)}, class0_num:{class0_num}")

    datas = ConcatDataset([datas_0, datas_1])

    print(f"all:{len(datas)}")

    # 各パラメータ作成
    device = torch.device('cuda')
    out_ft = 2

    
    model = resnet50(pretrained = False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, out_ft)
    model.to(device)

    train = torch.utils.data.DataLoader(
        datas,batch_size=16, shuffle=True,num_workers=4, pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    for epoch in range(num_epochs):
        for images, labels in tqdm(train):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        torch.save(model.state_dict(), f'./weights/updated_model_epoch{epoch+1}.pth')

if __name__ == "__main__":
    main()