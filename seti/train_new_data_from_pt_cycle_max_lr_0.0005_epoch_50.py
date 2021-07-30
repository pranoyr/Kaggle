#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from eff import EfficientNet
from vit_pytorch.vit import ViT
from torch.autograd import Variable
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


# train_annot = pd.read_csv(train_dir)
# train_annot = train_annot.values.tolist()
# print(train_annot[0])
# print(os.listdir(f"{root_dir}/0"))


def make_dataset(train_val_dist):
	# data = pd.read_csv(csv_file)
	data = train_val_dist.values.tolist()
	labels = np.array([i[1] for i in data])
	class_one_count = len(np.argwhere(labels==1).squeeze(1))
	class_zero_count = len(np.argwhere(labels==0).squeeze(1))

	class_counts = [class_zero_count, class_one_count]   # [0.1,1]
	num_samples = sum(class_counts)

	class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
	weights = [class_weights[labels[i]] for i in range(int(num_samples))]
	return data, weights


class SETIDataset(data.Dataset):
	'Characterizes a dataset for PyTorch'

	def __init__(self, root_dir, csv_file, num_classes=1, transform=None, image_set='train'):
		'Initialization'
		self.root_dir = root_dir
		self.transform = transform
		self.num_classes = num_classes
		self.image_set = image_set

		self.data, _ = make_dataset(csv_file)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data)

	def __getitem__(self, index):
		'Generates one sample of data'
		# ---- Get Inputs ----
		x = np.load(
			f"{self.root_dir}/{self.data[index][0][0]}/{self.data[index][0]}.npy")

		label = self.data[index][1]
		# if self.image_set=='train':
		# 	x = self.transform({"img":torch.from_numpy(x), "target":label})
		# else:
		if self.transform:	
			x = torch.from_numpy(x).view(3,-1,256)
			x = x.permute(1,2,0).numpy().astype('float32')
			x = self.transform(image=x)['image']
			x = x.type(torch.FloatTensor)
		else:
			x = torch.from_numpy(x).view(3,-1,256).type(torch.FloatTensor)
		# x = self.transform(image = x)
		

		# ---- Get Labels ----
		target = torch.tensor(label)
		return x, target.type(torch.FloatTensor)



def mixup_data(x, y, alpha=1.0, device="cuda"):
	'''Returns mixed inputs, pairs of targets, and lambda'''
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1

	batch_size = x.size()[0]
	# if use_cuda:
	index = torch.randperm(batch_size).to(device)
	# else:
	# 	index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam



def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a.unsqueeze(1)) + (1 - lam) * criterion(pred, y_b.unsqueeze(1))

class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res



class GridMask(DualTransform):
	
	def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
		super(GridMask, self).__init__(always_apply, p)
		if isinstance(num_grid, int):
			num_grid = (num_grid, num_grid)
		if isinstance(rotate, int):
			rotate = (-rotate, rotate)
		self.num_grid = num_grid
		self.fill_value = fill_value
		self.rotate = rotate
		self.mode = mode
		self.masks = None
		self.rand_h_max = []
		self.rand_w_max = []

	def init_masks(self, height, width):
		if self.masks is None:
			self.masks = []
			n_masks = self.num_grid[1] - self.num_grid[0] + 1
			for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
				grid_h = height / n_g
				grid_w = width / n_g
				this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
				for i in range(n_g + 1):
					for j in range(n_g + 1):
						this_mask[
							 int(i * grid_h) : int(i * grid_h + grid_h / 2),
							 int(j * grid_w) : int(j * grid_w + grid_w / 2)
						] = self.fill_value
						if self.mode == 2:
							this_mask[
								 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
								 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
							] = self.fill_value
				
				if self.mode == 1:
					this_mask = 1 - this_mask

				self.masks.append(this_mask)
				self.rand_h_max.append(grid_h)
				self.rand_w_max.append(grid_w)

	def apply(self, image, mask, rand_h, rand_w, angle, **params):
		h, w = image.shape[:2]
		mask = AF.rotate(mask, angle) if self.rotate[1] > 0 else mask
		mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
		image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
		return image

	def get_params_dependent_on_targets(self, params):
		img = params['image']
		height, width = img.shape[:2]
		self.init_masks(height, width)

		mid = np.random.randint(len(self.masks))
		mask = self.masks[mid]
		rand_h = np.random.randint(self.rand_h_max[mid])
		rand_w = np.random.randint(self.rand_w_max[mid])
		angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

		return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

	@property
	def targets_as_params(self):
		return ['image']

	def get_transform_init_args_names(self):
		return ('num_grid', 'fill_value', 'rotate', 'mode')


target_names = ['class 0', 'class 1']
class AverageMeter1:
	""" Computer precision/recall for multilabel classifcation

	Return:
			prec : Precision
			rec  : Recall
			avg  : Average Precision

	"""

	def __init__(self, name, fmt=':f', num_classes=1):
		# For each class
		self.name = name
		self.fmt = fmt
		# self.reset()
		self.precision = dict()
		self.recall = dict()
		self.average_precision = dict()
		self.gt = []
		self.y = []
		self.num_classes = num_classes
		self.o_list = []
		self.t_list = []

	def update(self, outputs, targets):
		o = torch.sigmoid(outputs)
		# o = torch.argmax(o, dim=1)
		
		self.o_list.extend(o.cpu().detach().numpy())
		self.t_list.extend(targets.cpu().numpy())

		# targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
		# self.y.append(outputs.detach().cpu())
		# self.gt.append(targets.detach().cpu())

		# preds = torch.cat(self.y)
		# targets = torch.cat(self.gt)
		# preds = preds.numpy()
		# targets = targets.numpy()


		# for i in range(self.num_classes):
		# 	self.precision[i], self.recall[i], _ = precision_recall_curve(
		# 		targets[:, i], preds[:, i])
		# 	self.average_precision[i] = average_precision_score(
		# 		targets[:, i], preds[:, i])

		# self.class_0 = self.average_precision[0]
		# self.class_1 = self.average_precision[1]

		# self.average_precision["micro"] = average_precision_score(targets, preds,
		# 														  average="micro")
		# self.avg = self.average_precision["micro"]

		self.avg = roc_auc_score(self.t_list, self.o_list)

	def __str__(self):
		# fmtstr = '{name} class0: {class_0' + self.fmt + '}, class1: ({class_1' + self.fmt + '})'
		fmtstr = '{name} {avg' + self.fmt + '}'
		out_array = np.array(self.o_list)
		mask_neg = np.array(self.o_list)<0.5
		mask_pos = np.array(self.o_list)>0.5
		out_array[mask_pos] = 1
		out_array[mask_neg] = 0

		# print(classification_report(np.array(self.t_list), out_array, target_names=target_names))
		# print(confusion_matrix(np.array(self.t_list), out_array))
		return fmtstr.format(**self.__dict__)

		


class AverageMeter2(object):
	"""Computes and stores the average and current value"""

	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


# In[ ]:



def val_epoch(model, data_loader, criterion, epoch, device):

	model.eval()

	metrics = AverageMeter1('ROC')
	losses = AverageMeter2('losses', ':.2f')
	progress = ProgressMeter(
		len(data_loader),
		[losses, metrics],
		prefix=f'Epoch {epoch}: ')
	# Training
	with torch.no_grad():
		for batch_idx, (data, targets) in enumerate(data_loader):
			# compute outputs
			data, targets = data.to(device), targets.to(device)
		
			outputs = model(data)
			loss = criterion(outputs, targets.unsqueeze(1))

			losses.update(loss.item(), data.size(0))
			metrics.update(outputs, targets)
		
		# show information
		print(f' * Val Loss {losses.avg:.3f}, Ap {metrics.avg:.3f}')
		return losses.avg, metrics.avg



def train_epoch(model, data_loader, criterion, optimizer, epoch, device, scheduler):

	model.train()

	metrics = AverageMeter1('ROC')
	losses = AverageMeter2('losses', ':.2f')
	progress = ProgressMeter(
		len(data_loader),
		[losses, metrics],
		prefix=f'Epoch {epoch}: ')
	iters = len(data_loader)
	# Training
	for batch_idx, (data, targets) in enumerate(data_loader):
		# compute outputs
		data, targets = data.to(device), targets.to(device)

		inputs, targets_a, targets_b, lam = mixup_data(data, targets,
													   0.2,  device)
		inputs, targets_a, targets_b = map(Variable, (inputs,
													  targets_a, targets_b))
		


	
		outputs = model(data)
		# loss = criterion(outputs, targets.unsqueeze(1))
		loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

		losses.update(loss.item(), data.size(0))
		metrics.update(outputs, targets)
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		# scheduler.step(epoch + batch_idx / iters)

		# show information
		if batch_idx % 10 == 0:
			progress.display(batch_idx)

	# show information
	print(f' * Train Loss {losses.avg:.3f}, Ap {metrics.avg:.3f}')
	return losses.avg, metrics.avg

# In[ ]:

def main():
	resume_path = './efficientnet_b0_ra-3dd342df.pth'
	start_epoch = 1
	wt_decay = 0.00001
	batch_size = 32

	# root_dir = '/home/neuroplex/Kaggle/seti/old_leaky_data/train_old'
	# train_csv = '/home/neuroplex/Kaggle/seti/old_leaky_data/train_labels_old.csv'
	root_dir = '/home/neuroplex/Kaggle/seti/train'
	train_csv = '/home/neuroplex/Kaggle/seti/train_labels.csv'

	# root_dir_old = '/home/neuroplex/Kaggle/seti/old_leaky_data/train_old'
	# train_csv_old = '/home/neuroplex/Kaggle/seti/old_leaky_data/train_labels_old.csv'

	df = pd.read_csv(train_csv)
	df['split'] = np.random.randn(df.shape[0], 1)
	msk = np.random.rand(len(df)) <= 0.9
	train_csv = df[msk]
	val_csv = df[~msk]


	# df = pd.read_csv(train_csv_old)
	# df['split'] = np.random.randn(df.shape[0], 1)
	# msk = np.random.rand(len(df)) <= 1.0
	# train_csv_old = df[msk]


	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")

	train_transform = A.Compose([
	A.Resize(256,256),
	A.OneOf([
	A.HorizontalFlip(p=0.5),
	A.VerticalFlip(p=0.5),
	A.ShiftScaleRotate(shift_limit= 0.2, scale_limit= 0.2, border_mode=0,
				rotate_limit= 20, value=0, mask_value=0),
	
	A.RandomResizedCrop(scale = [0.9, 1.0], p=1, height=512, width=512),
	
	]),
	ToTensorV2(p=1.0)
	])



	test_transform = A.Compose([
	A.Resize(256,256),
	# A.Normalize(mean=[-5.2037e-06, -1.4643e-04,  9.0275e-05], std = [0.9707, 0.9699, 0.9703], max_pixel_value=1, p=1.0),
	ToTensorV2(p=1.0)
	

	])

	_, weights = make_dataset(train_csv)
	training_data = SETIDataset(root_dir, train_csv, transform=train_transform, image_set = 'train')
	sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
	train_loader = torch.utils.data.DataLoader(training_data,
											batch_size=batch_size,
											sampler = sampler,
											num_workers=0)


	# _, weights = make_dataset(val_csv)
	validation_data = SETIDataset(root_dir, val_csv, transform=test_transform, image_set = 'val')
	# sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
	val_loader = torch.utils.data.DataLoader(validation_data,
											batch_size=batch_size,
											# sampler = sampler,
											num_workers=0)

	print(f'Number of training examples: {len(train_loader.dataset)}')
	print(f'Number of validation examples: {len(val_loader.dataset)}')
	import wandb
	wandb.login()
	default_config = {"scheduler":"onecycle","batch_size":32,
	"dataset":"new_data","model":"pretrained_imagenet","optimizer":"AdamW", "epochs":50, 
	"save_model_name":"seti_model_cycle_0.0007_50.pth"
	}
	wandb.init(name='train_new_data_from_pt_cycle_max_lr_0.0005_epoch_50_mix_up', 
		   project='Seti',
		   config=default_config,
		   entity='Pranoy')

	# tensorboard
	# summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs1')
	# define model
	# model = ResidualNet("ImageNet", 101, 1, "CBAM")
	# model = resnet101(num_classes=1)
	# model = ViT(
	# image_size = 256,
	# patch_size = 32,
	# num_classes = 1,
	# dim = 1024,
	# channels = 6,
	# depth = 6,
	# heads = 8,
	# mlp_dim = 2048,
	# dropout = 0.1,
	# emb_dropout = 0.1
	#)
	# model = vgg16(pretrained=False ,num_classes=1)
	model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)

	# if torch.cuda.device_count() > 1:
	# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	# model = nn.DataParallel(model)

	# if resume_path:
	# 	checkpoint = torch.load(resume_path)
	# 	model.load_state_dict(checkpoint['model_state_dict'])
	# 	epoch = checkpoint['epoch']
	# 	print("Model Restored from Epoch {}".format(epoch))
		# start_epoch = epoch + 1
	model.to(device)


	criterion = nn.BCEWithLogitsLoss()
	# from timm.optim import RAdam
	# optimizer = RAdam(model.parameters())
	# from timm.optim import AdamW
	# optimizer = torch.optim.AdamW(model.parameters())
	# from timm.optim import AdamW
	optimizer = torch.optim.AdamW(model.parameters())
	
	# from timm.scheduler import CosineLRScheduler
	from timm.scheduler import CosineLRScheduler
	# scheduler = CosineLRScheduler(optimizer, 100)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, steps_per_epoch=len(train_loader), epochs=50)

	th = -1
	# start training
	for epoch in range(start_epoch, 50):
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

			if (val_acc > th):
				state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()}
				torch.save(state, 'seti_model_cycle_0.0005_50.pth')
				print("Epoch {} model saved!\n".format(epoch))
				th = val_acc

		# scheduler.step(epoch)
				
if __name__=='__main__':
	main()