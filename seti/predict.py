import torch
from torch.optim import lr_scheduler
from torch.nn import BCEWithLogitsLoss
import numpy as np
import random

from sklearn.metrics import classification_report
import tensorboardX
import pandas as pd  
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


from train import ResidualNet

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


model = ResidualNet("ImageNet", 50, 1, "CBAM")
# model = nn.DataParallel(model)
# load pretrained weights
checkpoint = torch.load('./seti-model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

print("Model Restored")
model.eval()

  
# list1 = [["dsfsfds", 0.5]]
# df = pd.DataFrame(list1)
# df.to_csv('filename.csv', index=False)

resume_path = None
start_epoch = 1
wt_decay = 0.00001
batch_size = 32



# df = pd.read_csv(train_csv)
# df['split'] = np.random.randn(df.shape[0], 1)
# msk = np.random.rand(len(df)) <= 0.8
# train_csv = df[msk]
# val_csv = df[~msk]

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)




l = []
for filename in os.listdir('/home/cyberdome/Kaggle/seti/test/test'):
		file_path = '/home/cyberdome/Kaggle/seti/test/test/' + filename
		x = np.load(file_path)
		# x = transform(torch.from_numpy(x)).unsqueeze(0)
		x = torch.from_numpy(x).unsqueeze(0)
		# compute outputs
		x = x.type(torch.FloatTensor)
		x = x.to(device)
		outputs = model(x)
		# outputs = nn.Softmax(dim=1)(outputs)
		prob = torch.sigmoid(outputs).item()
		# if (prob > 0.5):
		# 	prob = 1
		# else:
		# 	prob = 0		# scores, indices = torch.topk(outputs, dim=1, k=1)
		# label = torch.argmax(outputs, dim=1).item()
		# if indices.item() == 0:
		# 	prob = 1 - scores.item()
		# else:
		# 	prob = scores.item()
		print(prob)
		l.append([filename.replace('.npy', ''), prob])
		
df = pd.DataFrame(l)
df.to_csv('submission.csv', index=False)