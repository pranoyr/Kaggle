#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from torch.optim import lr_scheduler
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score
import numpy as np
import random

from sklearn.metrics import classification_report
import tensorboardX
import argparse
from torchvision.models import resnet18
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

	def __init__(self, root_dir, csv_file, num_classes=1, transform=None):
		'Initialization'
		self.root_dir = root_dir
		self.transform = transform
		self.num_classes = num_classes

		self.data, _ = make_dataset(csv_file)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data)

	def __getitem__(self, index):
		'Generates one sample of data'
		# ---- Get Inputs ----
		x = np.load(
			f"{self.root_dir}/{self.data[index][0][0]}/{self.data[index][0]}.npy")
		x = self.transform(torch.from_numpy(x))
		# x = torch.from_numpy(x)

		# ---- Get Labels ----
		label = self.data[index][1]
		target = torch.tensor(label)
		return x.type(torch.FloatTensor), target.type(torch.LongTensor)


# root_dir = '/kaggle/input/seti-breakthrough-listen/train'
# train_csv = '/kaggle/input/seti-breakthrough-listen/train_labels.csv'
# a = SETIDataset(root_dir, train_csv)
# print(a.__getitem__(0)[0].shape)


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# tensor([ 1.1921e-06,  2.3842e-07,  1.2517e-06,  1.7881e-07,  1.4305e-06,
#         -1.1921e-07], device='cuda:0', dtype=torch.float16)
# tensor([0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0408], device='cuda:0',
#        dtype=torch.float16)


class BasicConv(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
		super(BasicConv, self).__init__()
		self.out_channels = out_planes
		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
							  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
		self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
								 momentum=0.01, affine=True) if bn else None
		self.relu = nn.ReLU() if relu else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
	def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
		super(ChannelGate, self).__init__()
		self.gate_channels = gate_channels
		self.mlp = nn.Sequential(
			Flatten(),
			nn.Linear(gate_channels, gate_channels // reduction_ratio),
			nn.ReLU(),
			nn.Linear(gate_channels // reduction_ratio, gate_channels)
		)
		self.pool_types = pool_types

	def forward(self, x):
		channel_att_sum = None
		for pool_type in self.pool_types:
			if pool_type == 'avg':
				avg_pool = F.avg_pool2d(
					x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
				channel_att_raw = self.mlp(avg_pool)
			elif pool_type == 'max':
				max_pool = F.max_pool2d(
					x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
				channel_att_raw = self.mlp(max_pool)
			elif pool_type == 'lp':
				lp_pool = F.lp_pool2d(
					x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
				channel_att_raw = self.mlp(lp_pool)
			elif pool_type == 'lse':
				# LSE pool only
				lse_pool = logsumexp_2d(x)
				channel_att_raw = self.mlp(lse_pool)

			if channel_att_sum is None:
				channel_att_sum = channel_att_raw
			else:
				channel_att_sum = channel_att_sum + channel_att_raw

		scale = torch.sigmoid(channel_att_sum).unsqueeze(
			2).unsqueeze(3).expand_as(x)
		return x * scale


def logsumexp_2d(tensor):
	tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
	s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
	outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
	return outputs


class ChannelPool(nn.Module):
	def forward(self, x):
		return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
	def __init__(self):
		super(SpatialGate, self).__init__()
		kernel_size = 7
		self.compress = ChannelPool()
		self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
			kernel_size-1) // 2, relu=False)

	def forward(self, x):
		x_compress = self.compress(x)
		x_out = self.spatial(x_compress)
		scale = torch.sigmoid(x_out)  # broadcasting
		return x * scale


class CBAM(nn.Module):
	def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
		super(CBAM, self).__init__()
		self.ChannelGate = ChannelGate(
			gate_channels, reduction_ratio, pool_types)
		self.no_spatial = no_spatial
		if not no_spatial:
			self.SpatialGate = SpatialGate()

	def forward(self, x):
		x_out = self.ChannelGate(x)
		if not self.no_spatial:
			x_out = self.SpatialGate(x_out)
		return x_out


# In[ ]:


# from bam import *


def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

		if use_cbam:
			self.cbam = CBAM(planes, 16)
		else:
			self.cbam = None

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		if not self.cbam is None:
			out = self.cbam(out)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

		if use_cbam:
			self.cbam = CBAM(planes * 4, 16)
		else:
			self.cbam = None

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		if not self.cbam is None:
			out = self.cbam(out)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, block, layers,  network_type, num_classes, att_type=None):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.network_type = network_type
		# different model config between ImageNet and CIFAR
		if network_type == "ImageNet":
			self.conv1 = nn.Conv2d(6, 64, kernel_size=7,
								   stride=2, padding=3, bias=False)
			self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
			self.avgpool = nn.AvgPool2d(7)
		else:
			self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
								   stride=1, padding=1, bias=False)

		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)

		if att_type == 'BAM':
			self.bam1 = BAM(64*block.expansion)
			self.bam2 = BAM(128*block.expansion)
			self.bam3 = BAM(256*block.expansion)
		else:
			self.bam1, self.bam2, self.bam3 = None, None, None

		self.layer1 = self._make_layer(
			block, 64,  layers[0], att_type=att_type)
		self.layer2 = self._make_layer(
			block, 128, layers[1], stride=2, att_type=att_type)
		self.layer3 = self._make_layer(
			block, 256, layers[2], stride=2, att_type=att_type)
		self.layer4 = self._make_layer(
			block, 512, layers[3], stride=2, att_type=att_type)

		self.fc = nn.Linear(512 * block.expansion, num_classes)
		# seltorch.sigmoid = nn.Sigmoid()

		init.kaiming_normal(self.fc.weight)
		for key in self.state_dict():
			if key.split('.')[-1] == "weight":
				if "conv" in key:
					init.kaiming_normal(self.state_dict()[key], mode='fan_out')
				if "bn" in key:
					if "SpatialGate" in key:
						self.state_dict()[key][...] = 0
					else:
						self.state_dict()[key][...] = 1
			elif key.split(".")[-1] == 'bias':
				self.state_dict()[key][...] = 0

	def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride,
					  downsample, use_cbam=att_type == 'CBAM'))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes,
						  use_cbam=att_type == 'CBAM'))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		if self.network_type == "ImageNet":
			x = self.maxpool(x)

		x = self.layer1(x)
		if not self.bam1 is None:
			x = self.bam1(x)

		x = self.layer2(x)
		if not self.bam2 is None:
			x = self.bam2(x)

		x = self.layer3(x)
		if not self.bam3 is None:
			x = self.bam3(x)

		x = self.layer4(x)

		if self.network_type == "ImageNet":
			x = self.avgpool(x)
		else:
			x = F.avg_pool2d(x, 4)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		# x = torch.sigmoid(x)
		return x


def ResidualNet(network_type, depth, num_classes, att_type):

	assert network_type in ["ImageNet", "CIFAR10",
							"CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
	assert depth in [18, 34, 50,
					 101], 'network depth should be 18, 34, 50 or 101'

	if depth == 18:
		model = ResNet(BasicBlock, [2, 2, 2, 2],
					   network_type, num_classes, att_type)

	elif depth == 34:
		model = ResNet(BasicBlock, [3, 4, 6, 3],
					   network_type, num_classes, att_type)

	elif depth == 50:
		model = ResNet(Bottleneck, [3, 4, 6, 3],
					   network_type, num_classes, att_type)

	elif depth == 101:
		model = ResNet(Bottleneck, [3, 4, 23, 3],
					   network_type, num_classes, att_type)

	return model


# if (__name__=='__main__'):
#     net = ResidualNet("ImageNet", 18, 2, att_type='CBAM')
#     x  = torch.randn(32, 6, 224,224)
#     output = net(x)
#     print(output.size())


# In[ ]:


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
		
		self.o_list.extend(o.cpu().numpy())
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
		fmtstr = '{name} class0: {class_0' + self.fmt + '}, class1: ({class_1' + self.fmt + '})'
		print(classification_report(np.array(self.t_list), np.array(self.o_list), target_names=target_names))
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

	metrics = AverageMeter1('Precision')
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
			loss = criterion(outputs, targets)

			losses.update(loss.item(), data.size(0))
			metrics.update(outputs, targets)
		
		# show information
		print(f' * Val Loss {losses.avg:.3f}, Ap {metrics.avg:.3f}')
		return losses.avg, metrics.avg


def train_epoch(model, data_loader, criterion, optimizer, epoch, device):

	model.train()

	metrics = AverageMeter1('Precision')
	losses = AverageMeter2('losses', ':.2f')
	progress = ProgressMeter(
		len(data_loader),
		[losses, metrics],
		prefix=f'Epoch {epoch}: ')
	# Training
	for batch_idx, (data, targets) in enumerate(data_loader):
		# compute outputs
		data, targets = data.to(device), targets.to(device)
	
		outputs = model(data)
		loss = criterion(outputs, targets.unsqeeze(0))

		losses.update(loss.item(), data.size(0))
		metrics.update(outputs, targets)
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# show information
		if batch_idx % 10 == 0:
			progress.display(batch_idx)

	# show information
	print(f' * Train Loss {losses.avg:.3f}, Ap {metrics.avg:.3f}')
	return losses.avg, metrics.avg


# In[ ]:

def main():
	resume_path = None
	start_epoch = 1
	wt_decay = 0.00001
	batch_size = 32

	# root_dir = '/kaggle/input/seti-breakthrough-listen/train'
	# train_csv = '/kaggle/input/seti-breakthrough-listen/train_labels.csv'
	root_dir = '/home/neuroplex/Kaggle/seti/train'
	train_csv = '/home/neuroplex/Kaggle/seti/train_labels.csv'

	df = pd.read_csv(train_csv)
	df['split'] = np.random.randn(df.shape[0], 1)
	msk = np.random.rand(len(df)) <= 0.8
	train_csv = df[msk]
	val_csv = df[~msk]


	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")

	transform = transforms.Compose([
		transforms.Normalize(mean=[1.1921e-06,  2.3842e-07,  1.2517e-06,  1.7881e-07,  1.4305e-06,
								-1.1921e-07], std=[0.0408, 0.0408, 0.0408, 0.0408, 0.0408, 0.0408])
	])


	_, weights = make_dataset(train_csv)
	training_data = SETIDataset(root_dir, train_csv, transform=transform)
	validation_data = SETIDataset(root_dir, val_csv, transform=transform)

	sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
	train_loader = torch.utils.data.DataLoader(training_data,
											batch_size=batch_size,
											sampler = sampler,
											num_workers=0)

	val_loader = torch.utils.data.DataLoader(validation_data,
											batch_size=batch_size,
											num_workers=0)

	print(f'Number of training examples: {len(train_loader.dataset)}')

	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')
	# define model
	model = ResidualNet("ImageNet", 101, 1, "CBAM")

	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
	# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	model = nn.DataParallel(model)

	if resume_path:
		checkpoint = torch.load(resume_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		epoch = checkpoint['epoch']
		print("Model Restored from Epoch {}".format(epoch))
		start_epoch = epoch + 1
	model.to(device)


	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)
	if resume_path:
		checkpoint = torch.load(resume_path)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			
	# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)

	th = 100000
	# start training
	for epoch in range(start_epoch, 100):
		# train, test model
		train_loss, train_acc = train_epoch(
			model, train_loader, criterion, optimizer, epoch, device)
		val_loss, val_acc = val_epoch(
			model, val_loader, criterion, epoch, device)
		lr = optimizer.param_groups[0]['lr']

		# saving weights to checkpoint
		if (epoch) % 1 == 0:
			# write summary
			summary_writer.add_scalar(
				'losses/train_loss', train_loss, global_step=epoch)
			summary_writer.add_scalar(
				'acc/train_acc', train_acc, global_step=epoch)
			summary_writer.add_scalar(
				'lr_rate', lr, global_step=epoch)

			summary_writer.add_scalar(
				'losses/val_loss', val_loss, global_step=epoch)
			summary_writer.add_scalar(
				'losses/val_acc', val_acc, global_step=epoch)

			state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()}
			
			torch.save(state, 'seti-model.pth')
			print("Epoch {} model saved!\n".format(epoch))
			
if __name__=='__main__':
	main()