import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import os
import cv2
import numpy as np
import torch
from torch import nn, einsum
from sklearn.metrics import plot_confusion_matrix

# from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from efficient_vit import EfficientViT
# from utils import transform_frame
import glob
from os import cpu_count
import json
from multiprocessing.pool import Pool
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
# from utils import custom_round, custom_video_round
from albumentations import Resize, Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate
# from .transforms.albu import IsotropicResize
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filter_type', type=str)
parser.add_argument('--mixing_type', type=str)
args = parser.parse_args()

filter_type = args.filter_type
mixing_type = args.mixing_type

config = './configs/architecture.yaml'
model_path = './efficient_vit.pth'

device = "cuda"

with open(config, 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
model = EfficientViT(config=config, channels=1280, selected_efficient_net=0)
model.load_state_dict(torch.load(model_path))
model.eval()
model = model.cuda()

# filter_type = 'filtered'
# mixing_type = 'lip'
method = 'efficient-vit'

parent_dir = '../../local-editing-dataset/experiment/test/{}/{}'.format(
    filter_type, mixing_type)

all_imgs_dir = [
    os.path.join(parent_dir, i) for i in os.listdir(parent_dir)
    if (i.endswith('.png') or i.endswith('.jpg'))
]

img_size = config['model']['image-size']

preds_list = []
imgs_path_list = []
for img_dir in tqdm(all_imgs_dir):
    frame = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
    resize = Resize(width=img_size, height=img_size)
    frame = resize(image=frame)['image']
    frame = torch.tensor(frame).to(device)
    with torch.no_grad():
        # predict for frame
        # channels first
        frame = frame.permute(2, 0, 1)
        frame = frame.float()
        prediction = model(frame.unsqueeze(0))
        preds = torch.sigmoid(prediction)
        preds_list.append(preds.cpu().numpy()[0][0])
        imgs_path_list.append(img_dir)

df = pd.DataFrame({'img_dir': imgs_path_list, 'pred': preds_list})
df.to_csv('{}_{}_{}.csv'.format(method, filter_type, mixing_type))