import torch
import torch.nn as nn
import warnings 
import torchvision.models as models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
warnings.filterwarnings(action='ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ConvNext(nn.Module): #백본 모델로 convNext 모델을 사용하였습니다. 해당 모델은 2022년 페이스북에서 발표한 CNN 계열 최신 모델입니다.
    def __init__(self, num_classes=2):
        super(ConvNext, self).__init__()
        self.backbone = models.convnext_large(pretrained=True)
        self.norm = nn.LayerNorm(1000)
        self.act = nn.SiLU()
        self.drop = nn.Dropout1d()
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.classifier(x)
        return x


train_transform = A.Compose([
      A.Resize(448,448),
      A.OneOf([
      A.Crop(x_min=0, y_min=0, x_max=224, y_max=224),  # Crop top-left quarter
      A.Crop(x_min=224, y_min=0, x_max=448, y_max=224),  # Crop top-right quarter
      A.Crop(x_min=0, y_min=224, x_max=224, y_max=448),  # Crop bottom-left quarter
      A.Crop(x_min=224, y_min=224, x_max=448, y_max=448),  # Crop bottom-right quarter
      ], p=1),

      A.OneOf([
      A.CLAHE(p=0.5),  # CLAHE algorithm
      A.GaussianBlur(p=0.5),    #Gaussian Blur
      ], p=0.5),
      A.OneOf([
          A.HorizontalFlip(p=1),  # Flip horizontally
          A.VerticalFlip(p=1),    # Flip vertically
      ], p=0.5),
      A.RandomRotate90(p=0.5),   # Randomly rotate 90 degrees
      A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
      A.Normalize(mean=0, std=1),
      ToTensorV2(),
])

classifier_transform=A.Compose([ #val, test셋에 대해서 offline crop 완료
      A.Resize(224,224),
    A.Normalize(mean=0, std=1),
    ToTensorV2(),
])

default_transform=A.Compose([
      A.Resize(640,640),
])

#해당 클래스는 torchvision 의 transform을 사용 시 정의하지 않아도 됩니다. 해당 예제에서는 albumentation을 활용하기 떄문에 필요합니다.
class Transforms:
      def __init__(self, transforms: A.Compose):
          self.transforms = transforms

      def __call__(self, img, *args, **kwargs):
          image=self.transforms(image=np.array(img))
          return image['image']