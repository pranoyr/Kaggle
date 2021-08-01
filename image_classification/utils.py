import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

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



