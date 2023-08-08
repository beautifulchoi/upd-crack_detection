import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import warnings
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import Transforms, train_transform, classifier_transform, ConvNext
from config import CFG,seed_everything
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim

warnings.filterwarnings(action='ignore')
root_dir="/content/drive/MyDrive/classifier/align" #root dir of dataset
save_dir="/content/drive/MyDrive/classifier/" #ckpt saved dir 
pretrained_ckpt="/content/drive/MyDrive/classifier/best_0725_crop.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#dataset and dataloader
train_dataset=ImageFolder(root=root_dir+'/train', transform=Transforms(train_transform))
val_dataset=ImageFolder(root=root_dir+'/val', transform=Transforms(transforms=classifier_transform))
test_dataset=ImageFolder(root=root_dir+'/test', transform=Transforms(transforms=classifier_transform))

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=12)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=12)
test_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=12)

#model and optim, scheduler
model=ConvNext(num_classes=2)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Training loop
def train(model, optimizer, train_loader, val_loader, scheduler, device, pretrained_ckpt=False):
    if pretrained_ckpt:
      model.load_state_dict(pretrained_ckpt)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    cnt=0
    best_score = 0
    best_model = None

    for epoch in tqdm(range(1, CFG['EPOCHS']+1)):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_score < _val_score:
            best_score = _val_score
            best_model = model
            torch.save(best_model.state_dict(), save_dir+'/best_0727_crop.pt')
            cnt=0
        else:
          print(f"early stopping count: {cnt+1}")
          cnt+=1
        if (cnt==10):
          print(f"early stop. epoch: {epoch}")
          return best_model

    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            pred = model(imgs)

            loss = criterion(pred, labels)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()

            val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average='weighted')

    return _val_loss, _val_score



if __name__=="__main__":
    seed_everything(CFG['SEED']) # Seed 고정
    pretrained_ckpt=torch.load(pretrained_ckpt)
    infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device, pretrained_ckpt)
    