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
