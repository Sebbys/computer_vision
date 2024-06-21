import torchvision
from torchvision.models.detection import *
import torch
from torch.utils.data import *
from tqdm import tqdm
from utils.data import getData
from torchvision.models.detection import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def getModel(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    return model

def collate_fn(batch):
    return batch

BATCH_SIZE = 4
EPOCH = 5
MOMENTUM = 0.0009
LEARNING_RATE = 0.009
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = getModel(8).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM)

train_loader = DataLoader(getData(f='./train.csv'), batch_size = BATCH_SIZE, shuffle=True)
test_loader = DataLoader(getData(f='./test.csv'), batch_size = BATCH_SIZE, shuffle=True)

for epoch in range(EPOCH):
    model.train()
    print('Epoch: ', epoch + 1)
    for batch, (src,target) in enumerate(tqdm(train_loader, desc=f'epoch {epoch+1}/{EPOCH}')):
        src = torch.permute(src, (0, 3, 1, 2)).to(DEVICE)
        box,lab,targets = [],[],[]
        for i in range(BATCH_SIZE):
            b = []
            for j in range(4):
                b.append(target[0][j][i])
            box.append([b])
            lab.append([target[1][0][i]])

        box = torch.tensor(box, dtype=torch.float32).to(DEVICE)
        lab = torch.tensor(lab, dtype=torch.int64).to(DEVICE)

        for i in range(len(src)):
            d = {}
            d['boxes']=box[i]
            d['labels']=lab[i]
            targets.append(d)

        loss_dict = model(src,targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.step()
        optimizer.zero_grad()
        
    print("Epoch: %d, Loss: %f" % (epoch + 1, float(losses)))

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch, (images, targets) in enumerate(tqdm(data_loader, desc='Evaluating')):
            images = torch.permute(images, (0, 3, 1, 2)).to(device)
            
            boxes, labels = [], []
            for i in range(len(images)):
                b = []
                for j in range(4):
                    b.append(targets[0][j][i])
                boxes.append([b])
                labels.append([targets[1][0][i]])

            boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.int64).to(device)

            targets = []
            for i in range(len(images)):
                d = {}
                d['boxes'] = boxes[i]
                d['labels'] = labels[i]
                targets.append(d)

            # During evaluation, the model returns predictions, not losses
            predictions = model(images)

            # Calculate a simple loss based on bounding box predictions
            loss = 0
            for pred, target in zip(predictions, targets):
                # Calculate loss for bounding boxes
                if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
                    box_loss = torch.nn.functional.mse_loss(pred['boxes'], target['boxes'])
                    loss += box_loss

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

# After training loop
print("Evaluating model performance...")
test_loss = evaluate(model, test_loader, DEVICE)
print(f"Test Loss: {test_loss:.4f}")