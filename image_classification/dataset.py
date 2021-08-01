from datasets.vrd import VRDDataset
from datasets.custom_vrd import CustomDataset
from datasets.vg import VGDataset
from datasets.seti import SETIDataset, make_dataset
import torch
import pandas as pd

def get_training_data(opt):
	if opt.dataset == 'seti_leaky_data':
    	train_csv = pd.read_csv(train_csv)

		# df['split'] = np.random.randn(df.shape[0], 1)
		# msk = np.random.rand(len(df)) <= 1.0
		# train_csv_old = df[msk]
		_, weights = make_dataset(pd.concat(opt.train_csv)
		training_data = SETIDataset(root_dir, train_csv, transform=train_transform, image_set = 'train')
		sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
		train_loader = torch.utils.data.DataLoader(training_data,
												batch_size=batch_size,
												sampler = sampler,
												num_workers=0

		return training_data


def get_validation_data(opt):
	if opt.dataset == 'seti_leaky_data':
    		
		test_csv = pd.read_csv(test_csv)
		# _, weights = make_dataset(val_csv)
		validation_data = SETIDataset(root_dir, val_csv, transform=test_transform, image_set = 'val')
		# sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
		val_loader = torch.utils.data.DataLoader(validation_data,
												batch_size=batch_size,
												# sampler = sampler,
												num_workers=0)



	return validation_data


def get_train_val_data(d_ratio=0.8):
	if opt.dataset == 'seti_leaky_data':
    	train_csv = pd.read_csv(train_csv)
		# df['split'] = np.random.randn(df.shape[0], 1)
		# msk = np.random.rand(len(df)) <= 1.0
		# train_csv_old = df[msk]
		_, weights = make_dataset(pd.concat(opt.train_csv)
		training_data = SETIDataset(root_dir, train_csv, transform=train_transform, image_set = 'train')
		sampler = data.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))
		train_loader = torch.utils.data.DataLoader(training_data,
												batch_size=batch_size,
												sampler = sampler,
												num_workers=0

		return training_data

	
