from albumentations.pytorch.transforms import ToTensorV2

import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import pandas as pd
from utils import *



def train_epoch(model, data_loader, criterion, optimizer, epoch, device):

	model.train()

	metrics = AverageMeter1('ROC')
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
		loss = criterion(outputs, targets.unsqueeze(1))

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
