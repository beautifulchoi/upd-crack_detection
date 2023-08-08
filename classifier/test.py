import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from model import ConvNext
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import classifier_transform, Transforms
from config import CFG,seed_everything
from matplotlib import pyplot as plt

root_dir="/content/drive/MyDrive/classifier/align"
ckpt= torch.load("/content/drive/MyDrive/classifier/best_0725_crop.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#test dset
test_dataset=ImageFolder(root=root_dir+'/test', transform=Transforms(transforms=classifier_transform))
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=12)

#모델 테스트
def test(model,device, test_loader):
  model.eval()

  preds, true_labels = [], []
  target_names = ['negative', 'positive'] #0: neg , 1: pos // neg가 더 많음(pos 는 분할 후 크랙 없는 쪽 지워버렸음)
  with torch.no_grad():
    for imgs, labels in tqdm(iter(test_loader)):
        imgs = imgs.float().to(device)
        labels = labels.to(device)

        pred = model(imgs)

        preds += pred.argmax(1).detach().cpu().numpy().tolist()
        true_labels += labels.detach().cpu().numpy().tolist()

        for i in range(len(pred)):
          if pred.argmax(1)[i]!=labels[i]:
            plt.imshow(imgs[i].detach().cpu().permute(1,2,0))
            plt.title(f'pred:{pred.argmax(1)[i]}, true:{labels[i]}')
            plt.show()

    result=classification_report(true_labels, preds, target_names=target_names)
    print(result)
  return result

if __name__=="__main__":
    seed_everything(CFG['SEED']) # Seed 고정
    model=ConvNext(num_classes=2)
    model.load_state_dict(ckpt)
    model=model.to(device)
    result=test(model, device, test_loader)
