import torch
from torchvision import models, transforms
from torch.utils.data import *
from glob import glob
from tqdm import tqdm
from PIL import Image
import pandas as pd
from sklearn import metrics
import math
import numpy as np
import os
import datetime

class Mytestdatasets(Dataset):
    def __init__(self, datasets, transform):
        self.transformer = transform
        self.dirname = datasets
        self.files = glob(f"{self.dirname}/*")

    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        image = self.transformer(image)
        file_name = os.path.basename(self.files[idx])
        return image, file_name


def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def main():
    out_ft = 2
    device = torch.device('cuda')

    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, out_ft)
    model.to(device)
    model.load_state_dict(torch.load('weights/updated_model_epoch20.pth'))
    model.eval()


    nonTC = "nonTC"
    TC = "TC"

    dataset_test_path = [
        f"./test_1/test/",
        f"./test_2/test/",
        f"./test_3/test/",
    ]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 224x224にリサイズ
        transforms.ToTensor(),          # テンソルに変換
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_data = []
    for datasets in dataset_test_path:
        set = Mytestdatasets(datasets, transform)
        test_data.append(set)
    test_datas = ConcatDataset(test_data)
    
    tests = torch.utils.data.DataLoader(
        test_datas,batch_size=1)
    
    preds = []
    files = []
    with torch.no_grad():
        for image, file_name in tqdm(tests):
            image = image.to(device)
            pred = model(image)
            print(pred)
            y = torch.argmax(pred, axis=-1)
            print(y)
            preds.append(y.tolist())
            file_name = list(file_name)
            files.append(file_name)

    preds_flat = flatten(preds)
    files_flat = flatten(files)
    
    matrix = pd.DataFrame([files_flat, 
                          preds_flat]).transpose()
    
    matrix.to_csv(f'result/out_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.tsv', header = None, index = None, sep='\t')

if __name__ == "__main__":
    main()