from albumentations.pytorch.transforms import ToTensorV2
import numpy as np  # linear algebra
import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils import data
import torch


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
