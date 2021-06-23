# # class AverageMeter(object):
# #     """Computes and stores the average and current value"""

# #     def __init__(self, name, fmt=':f'):
# #         self.name = name
# #         self.fmt = fmt
# #         self.reset()

# #     def reset(self):
# #         self.val = 0
# #         self.avg = 0
# #         self.sum = 0
# #         self.count = 0

# #     def update(self, val, n=1):
# #         self.val = val
# #         self.sum += val * n
# #         self.count += n
# #         self.avg = self.sum / self.count

# #     def __str__(self):
# #         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
# #         print(self.__dict__)

# #         print(fmtstr)
# #         return fmtstr.format(**self.__dict__)


# # avg = AverageMeter("loss", ':.2')
# # avg.update(0.1, 32)
# # avg.update(1.2, 32)
# # print(avg)



# # a = [1,2,3]
# # a.extend([1,2,3.4])

# # print(a)




# # import torch

# # target = torch.tensor([2,4,3])
# # one_hot = torch.nn.functional.one_hot(target, num_classes=5)

# # print(one_hot)

# # import numpy as np
# # l = np.array([1,1,0,0,0,1,1,1,0])
# # a = np.argwhere(l==1).squeeze(1)
# # print(a)


# # from random import betavariate
# # from torch.utils import data


# # class SETIDataset(data.Dataset):
# # 	'Characterizes a dataset for PyTorch'

# # 	def __init__(self, num_classes=2, transform=None):
# # 		'Initialization'
# # 		a = 1
# # 		# self.data, _ = make_dataset(csv_file)

# # 	def __len__(self):
# # 		'Denotes the total number of samples'
# # 		return 220

# # 	def __getitem__(self, index):
# # 		'Generates one sample of data'
# # 		# ---- Get Inputs ----
# # 		print("#$%#$")
# # 		print(index)
# # 		return torch.tensor([index]), torch.tensor([1])

# import torch
# from torch.utils.data import BatchSampler, SequentialSampler, DataLoader, Dataset, WeightedRandomSampler
# # # a = list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
# # # print(a)


# # training_data = SETIDataset()
# # # validation_data = get_validation_set(opt, test_transform)

# # # a = SequentialSampler(training_data)
# # # print(a)

# # # for i in a:
# # #     print(i)

# # sampler = torch.utils.data.sampler.BatchSampler(
# # 	torch.utils.data.sampler.SequentialSampler(training_data),
# # 	batch_size=10,
# # 	drop_last=False)

# # train_loader = torch.utils.data.DataLoader(training_data,
# # 										   batch_size=10,
# # 										   sampler = sampler,
# # 										   num_workers=0)

# # for  i in train_loader:
# # 	print(i[0].shape)
# # 	break
	



# class MyDataset(Dataset):
# 	def __init__(self):
# 		self.data = torch.Tensor(100,3,32,32)
		
# 	def __getitem__(self, index):
# 		print(index)
# 		x = self.data[index]
# 		return x
	
# 	def __len__(self):
# 		return len(self.data)

# dataset = MyDataset()        
# sampler = torch.utils.data.sampler.BatchSampler(
# 	torch.utils.data.sampler.RandomSampler(dataset),
# 	batch_size=20,
# 	drop_last=False)

# sampler =  torch.utils.data.sampler.RandomSampler(dataset)
# loader = DataLoader(
# 	dataset,
# 	sampler=sampler, batch_size=20)

# # print(len(list(sampler)))

# # for data in loader:
# # 	# print(data.shape)
# # 	break



# # dataset = MyDataset()        

# # sampler =  torch.utils.data.sampler.WeightedRandomSampler(dataset)
# # loader = DataLoader(
# # 	dataset,
# # 	sampler=sampler, batch_size=20)

# # for data in loader:
# # 	print(data.shape)
# # 	break

# import pandas as pd
# import numpy as np

# df = pd.read_csv('/Users/pranoyr/Downloads/train_labels.csv')
# df['split'] = np.random.randn(df.shape[0], 1)

# msk = np.random.rand(len(df)) <= 0.8

# train = df[msk]
# test = df[~msk]
# for i in [train, test]:
# 	data = i.values.tolist()
# 	labels = np.array([i[1] for i in data])
# 	class_one_count = len(np.argwhere(labels==1).squeeze(1))
# 	class_zero_count = len(np.argwhere(labels==0).squeeze(1))

# 	print(class_one_count)
# 	print(class_zero_count)
# 	print()


from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# X, y = load_breast_cancer(return_X_y=True)
# clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
# roc_auc_score(y, clf.predict_proba(X)[:, 1])

print(roc_auc_score([0,1,1,1], [0.2,0.7,0.5,0.9]))