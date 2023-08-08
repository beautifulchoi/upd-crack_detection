import warnings
from ultralytics import YOLO
import torch


warnings.filterwarnings("ignore")

SEED = 2024
BATCH_SIZE = 16
MODEL = "/content/drive/MyDrive/yoloV8학습/crack_detection_subset_auto_annotation"
PRETRAINED_PATH="/content/drive/MyDrive/yoloV8학습/crack_detection_subset_auto_annotation/train/weights/best.pt"
YAML_PATH="/content/drive/MyDrive/yoloV8학습/rebuild_with_pseudo_annotation/data.yaml"

def train():

    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')


    model = YOLO(PRETRAINED_PATH, device=device)
    results = model.train(
        data=YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=BATCH_SIZE,
        patience=15,
        save_period=10,
        pretrained=True,
        workers=16,
        device=0,
        exist_ok=True,
        resume=True,
        project=f"{MODEL}",
        name="train",
        seed=SEED,
        optimizer="RAdam",
        lr0=1e-3,
        val=True,
        cache='ram',
        mixup = 0.5   # use mix-up Augmentation
        )

if __name__=="__main__":
    train()