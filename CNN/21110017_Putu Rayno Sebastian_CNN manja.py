import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



class Data(Dataset):
    def __init__(self, f="./batik"):

        self.fitur, self.label = [], []

        for i in range(1, 50):
            par = cv.imread("./batik/batik-parang/" + str(i) + ".jpg")
            if par is None:
                print(f"Failed to load ./batik/batik-parang/{i}.jpg. Skipping...")
                continue

            gar = cv.imread("./batik/batik-garutan/" + str(i) + ".jpg")
            if gar is None:
                print(f"Failed to load ./batik/batik-garutan/{i}.jpg. Skipping...")
                continue

            # print(gar)

            # if par != null:
            par = cv.resize(par, (500, 500))
            gar = cv.resize(gar, (500, 500))

            par = (par - np.min(par)) / np.ptp(par)
            gar = (gar - np.min(gar)) / np.ptp(gar)

            self.fitur.append(par)
            self.label.append(0)
            self.fitur.append(gar)
            self.label.append(1)  

    def __getitem__(self, item):
        fitur, label = self.fitur[item], self.label[item]
        return torch.tensor(fitur, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.fitur)

class ConvNet(nn.Module):
    def __init__(self, input=1000, output=2, hidden=128, dropout=0.1):
        super(ConvNet, self).__init__()

        self.convblock1 = nn.Sequential(
                            nn.Conv2d(3, 32, kernel_size=(3,3), padding=1),
                            nn.ReLU(),
                            )

        self.convblock2 = nn.Sequential(
                            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d((2,2)),
                            )
        self.convblock3 = nn.Sequential(
                            nn.Conv2d(32, 10, kernel_size=(3,3), padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d((3,3)),
                            )
        self.convblock4 = nn.Sequential(
                            nn.Conv2d(10, 2, kernel_size=(3,3), padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d((3,3)),
                            )
        self.flat = nn.Flatten()

        self.MLP = nn.Sequential(
                            nn.Linear(1458, 400),
                            nn.ReLU(),
                            nn.Linear(400, 2),
                            nn.ReLU(),
                            )
        self.drop = nn.Dropout(dropout)
        self.soft = nn.Softmax()


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        
        x = self.flat(x)

        x = self.MLP(x)
        x = self.drop(x)
        x = self.soft(x)

        return x


def main():
    BATCH_SIZE = 8
    EPOCH = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Data()
    train_len = int(len(dataset) * 0.7)
    test_len = len(dataset) - train_len

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ConvNet(input=1000, output=2, hidden=128, dropout=0.1)
    model = model.to(device)  

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    true_labels = []
    pred_labels = []

    for epoch in range(EPOCH):
        loss_all = 0
        for batch, (fitur, label) in enumerate(train_loader):
            fitur = torch.permute(fitur, (0, 3, 1, 2))
            fitur = fitur.to(device)
            label = label.to(device)
            pred = model(fitur)
            loss = loss_fn(pred, label)
            loss_all += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"loss = {loss_all/BATCH_SIZE} || epoch : {epoch}")

    model.eval()
    with torch.no_grad():
        for fitur, label in test_loader:
            fitur = torch.permute(fitur, (0, 3, 1, 2))
            fitur = fitur.to(device)
            label = label.to(device)
            pred = model(fitur)
            _, predicted = torch.max(pred, 1)
            true_labels.extend(label.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    cm = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
                

if __name__=="__main__":
    main()