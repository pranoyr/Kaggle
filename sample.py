class AverageMeter(object):
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
        print(self.__dict__)

        print(fmtstr)
        return fmtstr.format(**self.__dict__)


avg = AverageMeter("loss", ':.2')
avg.update(0.1, 32)
avg.update(1.2, 32)
print(avg)


import torch

target = torch.tensor([2,4,3])
one_hot = torch.nn.functional.one_hot(target, num_classes=5)

print(one_hot)




