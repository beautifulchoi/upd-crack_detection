import random
import os
import torch
import numpy as np

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':100,
    'LEARNING_RATE':1e-4,
    'BATCH_SIZE':8,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
