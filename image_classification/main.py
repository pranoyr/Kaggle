

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
import wandb
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
from opts import parse_opts
from train import train_epoch
from validate import val_epoch


def get_scheduler(opt, optimizer):
	if opt.scheduler == "plateau":
		scheduler = lr_scheduler.ReduceLROnPlateau(
			optimizer, 'min', patience=opt.patience)
	elif opt.scheduler == "multi_step":
		scheduler = lr_scheduler.MultiStepLR(
			optimizer, milestones=[20, 40])
	elif opt.scheduler == "step_lr":
		scheduler = lr_scheduler.StepLR(
			optimizer, step_size=5, gamma=0.1, last_epoch=-1)
	return scheduler

def get_optimizer(opt, model):
	if opt.optimizer == "adam":
		return optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


def main():
	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	opt = parse_opts()

	opt.train_root_dir_leaky = '/home/neuroplex/Kaggle/seti/old_leaky_data/train_old'
	opt.train_csv_leaky = '/home/neuroplex/Kaggle/seti/old_leaky_data/train_labels_old.csv'

	opt.test_root_dir_leaky = '/home/neuroplex/Kaggle/seti/old_leaky_data/test_old'
	opt.test_csv_leaky = '/home/neuroplex/Kaggle/seti/old_leaky_data/test_labels_old.csv'

	opt.train_root_dir_new = '/home/neuroplex/Kaggle/seti/old_leaky_data/train'
	opt.train_csv_new = '/home/neuroplex/Kaggle/seti/old_leaky_data/train_labels.csv'


	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:1" if use_cuda else "cpu")


	configs = {"device":opt.device,
				"lr_scheduler": opt.scheduler,
				"batch_size":opt.batch_size,
				"epochs":opt.epochs,
				"n_classes":opt.classes,
				"resume_path":opt.resume_path
	}

	train_transform = A.Compose([
	A.Resize(256,256),
	A.HorizontalFlip(p=0.5),
	A.VerticalFlip(p=0.5),
	A.Transpose(),
	A.ShiftScaleRotate(),	
	A.RandomRotate90(),
	# A.GridDropout( holes_number_x=5, holes_number_y=5)
	A.GridDropout(),
	# A.Normalize(mean=[-5.2037e-06, -1.4643e-04,  9.0275e-05], std = [0.9707, 0.9699, 0.9703], max_pixel_value=1, p=1.0),
	ToTensorV2(p=1.0)])

	test_transform = A.Compose([
	A.Resize(256,256),
	# A.Normalize(mean=[-5.2037e-06, -1.4643e-04,  9.0275e-05], std = [0.9707, 0.9699, 0.9703], max_pixel_value=1, p=1.0),
	ToTensorV2(p=1.0)])


	wandb.login()
	wandb.init(name=opt.model+opt.dataset, 
		   project='Seti',
		   config = configs,
		   entity='Pranoy')


	train_loader = get_train_loader(opt, train_transform)
	val_loader = get_validation_loader(opt, test_transform)
	print(f'Number of training examples: {len(train_loader.dataset)}')
	print(f'Number of training examples: {len(val_loader.dataset)}')

	model = get_model(opt)
	if opt.resume_path:
		checkpoint = torch.load(opt.resume_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		epoch = checkpoint['epoch']
		print("Model Restored from Epoch {}".format(epoch))
		start_epoch = epoch + 1
	# if torch.cuda.device_count() > 1 and opt.multi_gpu:
	# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# 	dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	# 	model = nn.DataParallel(model)
	model.to(device)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = get_optimizer(opt)
	# if opt.resume_path:
	# 	checkpoint = torch.load(opt.resume_path)
	# 	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	scheduler = get_scheduler(opt, optimizer)
	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

	th = -1
	# start training
	for epoch in range(start_epoch, 1000):
		# train, test model
		train_loss, train_acc = train_epoch(
			model, train_loader, criterion, optimizer, epoch, device, scheduler)

		# validate
		if (epoch) % 1 == 0:
			val_loss, val_acc = val_epoch(model, val_loader, criterion, epoch, device)
			lr = optimizer.param_groups[0]['lr']
			
			wandb.log({
				"Epoch": epoch,
				"Train Loss": train_loss,
				"Train Acc": train_acc,
				"Valid Loss": val_loss,
				"Valid Acc": val_acc,
				"lr":lr})

			if opt.scheduler == 'plateau':
				scheduler.step(val_loss)

			if (val_acc > th):
				state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()}
				
				torch.save(state, opt.model + opt.dataset + ".pth")
				print("Epoch {} model saved!\n".format(epoch))
				th = val_acc
				
if __name__=='__main__':
	main()