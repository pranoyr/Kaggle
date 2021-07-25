#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from eff import EfficientNet
from vit_pytorch.vit import ViT
from torch.optim import lr_scheduler
from albumentations.augmentations import functional as AF
from albumentations.core.transforms_interface import DualTransform
from torch.nn import BCEWithLogitsLoss
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from transforms import RandomHorizontalFlip, RandomVerticalFlip
from albumentations.pytorch.transforms import ToTensorV2
from transforms import GaussianNoise
from vgg import vgg16
import numpy as np
import random
import albumentations

from sklearn.metrics import classification_report
import tensorboardX
import argparse
from torchvision.models import resnet101
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils import data
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score



	# model = vgg16(pretrained=False ,num_classes=1)
model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=1)

	# if torch.cuda.device_count() > 1:
	# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	# model = nn.DataParallel(model)

	# if resume_path:
	# 	checkpoint = torch.load(resume_path)
	# 	model.load_state_dict(checkpoint['model_state_dict'])
	# 	epoch = checkpoint['epoch']
	# 	print("Model Restored from Epoch {}".format(epoch))
	# 	# start_epoch = epoch + 1
	# model.to(device)


# criterion = nn.BCEWithLogitsLoss()
from timm.optim import RAdam
optimizer = RAdam(model.parameters())

from timm.scheduler import CosineLRScheduler
scheduler = CosineLRScheduler(optimizer, 10)
# scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10)

th = -1
# start training
for epoch in range(1,30):
	# train, test model
	# train_loss, train_acc = train_epoch(
	# model, train_loader, criterion, optimizer, epoch, device, scheduler)

	# # validate
	# if (epoch) % 1 == 0:
	# val_loss, val_acc = val_epoch(model, val_loader, criterion, epoch, device)

	lr = optimizer.param_groups[0]['lr']
	print(lr,epoch)
	print("********")

	scheduler.step(epoch)

	

	

	# wandb.log({
	# 	"Epoch": epoch,
	# 	"Train Loss": train_loss,
	# 	"Train Acc": train_acc,
	# 	"Valid Loss": val_loss,
	# 	"Valid Acc": val_acc,
	# 	"lr":lr})




	# if (val_acc > th):
	# 	state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
	# 			'optimizer_state_dict': optimizer.state_dict()}
		
	# 	torch.save(state, 'new_data_model_from_pt.pth')
	# 	print("Epoch {} model saved!\n".format(epoch))
	# 	th = val_acc
		