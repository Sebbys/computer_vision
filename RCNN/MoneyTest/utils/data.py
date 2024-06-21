from torchvision import transforms as T
import pandas as pd
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset

class getData(Dataset):
    def __init__(self, f='./train.csv'):
        self.src, self.trg = [], []

        self.Transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.RandomHorizontalFlip(0.5),
            T.RandomCrop(0.2),
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ])

        data = pd.read_csv(f)
        for row in range(len(data)):
            box = [data['xmin'][row], data['ymin'][row], data['xmax'][row], data['ymax'][row]]
            lab = [data['name'][row]]

            lab = list(map(self.mapping, lab))
            img = cv.imread(data['filename'][row])
            img = (img - np.min(img)) / (np.ptp(img))

            self.src.append(img)
            self.trg.append([box, lab])

    def __len__(self):
        return len(self.trg)
    
    #usage 
    def mapping(self, x):
        if x == 1000:
            return 0
        elif x == 2000:
            return 1
        elif x == 5000:
            return 2
        elif x == 10000:
            return 3
        elif x == 20000:
            return 4
        elif x == 50000:
            return 5
        elif x == 75000:
            return 6
        elif x == 100000:
            return 7

    def __getitem__(self, index):
        return torch.tensor(self.src[index], dtype=torch.float32), self.trg[index]
