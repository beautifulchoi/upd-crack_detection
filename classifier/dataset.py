import glob
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import warnings
from torchvision.datasets import ImageFolder
from glob import glob
from torch.utils.data import Dataset
from config import CFG


warnings.filterwarnings(action='ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class CustomDataset(Dataset): # deprecated
    def __init__(self, root_dir, split="train", transforms=None):
        self.transforms = transforms
        neg_dir, pos_dir= glob(root_dir+'/*')
        print(pos_dir)
        pos_imgs=glob(pos_dir+'/*')
        neg_imgs=glob(neg_dir+'/*')
        pos_train,pos_test = train_test_split(pos_imgs, test_size=0.1, random_state=CFG['SEED'])
        pos_train,pos_val = train_test_split(pos_train, test_size=0.1, random_state=CFG['SEED'])
        neg_train,neg_test = train_test_split(neg_imgs, test_size=0.1, random_state=CFG['SEED'])
        neg_train,neg_val = train_test_split(neg_train, test_size=0.1, random_state=CFG['SEED'])
        if(split=='train'):
          self.pos_path=pos_train
          self.neg_path=neg_train
        elif(split=='val'):
          self.pos_path=pos_val
          self.neg_path=neg_val
        else:
          self.pos_path= pos_test
          self.neg_path=neg_test


    def __getitem__(self, index):
        imgs=self.pos_path+self.neg_path
        if index<len(self.pos_path):
          label=0 #pos =0
        else:
          label=1 #neg =1

        img_path = imgs[index]
        image = cv2.imread(img_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']


        return image, label


    def __len__(self):
        return len(self.pos_path+self.neg_path)


