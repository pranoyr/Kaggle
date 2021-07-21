import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd


from torch.utils import data
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


mean = 0.
std = 0.
nb_samples = 0.

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



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



root_dir = '/home/cyberdome/Kaggle/seti/train'
train_csv = '/home/cyberdome/Kaggle/seti/train_labels.csv'

train_transform = A.Compose([
	A.Resize(256,256),
	ToTensorV2(p=1.0)

	])

df = pd.read_csv(train_csv)
df['split'] = np.random.randn(df.shape[0], 1)
msk = np.random.rand(len(df)) <= 1.0
train_csv = df[msk]
val_csv = df[~msk]

train_data = SETIDataset(root_dir, train_csv, transform=train_transform, image_set = 'val')
train_loader = torch.utils.data.DataLoader(train_data,
											batch_size=32,
											# sampler = sampler,
											num_workers=0)

def _resize_image_and_masks(image, self_min_size=800, self_max_size=1333):
	im_shape = torch.tensor(image.shape[-2:])
	min_size = float(torch.min(im_shape))
	max_size = float(torch.max(im_shape))
	scale_factor = self_min_size / min_size
	if max_size * scale_factor > self_max_size:
		scale_factor = self_max_size / max_size
	image = torch.nn.functional.interpolate(
		image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True,
		align_corners=False)[0]
	return image
	   

for data in train_loader:
	images, targets = data
	# images = images[0]
	# images = _resize_image_and_masks(images).unsqueeze(0)
	# images = images.view(images.size(0), images.size(1), -1)
	mean += images.mean(2).sum(0)
	std += images.std(2).sum(0)
	nb_samples += images.size(0)

mean /= nb_samples
std /= nb_samples

print(mean)
print(std)

	
