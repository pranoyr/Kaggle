import torch
from torch import nn

from models.effnet import EfficientNet


def generate_model(opt):
    if opt.model == 'efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=opt.num_classes)
    return model




# def make_data_parallel(model, is_distributed, device):
#     if is_distributed:
#         if device.type == 'cuda' and device.index is not None:
#             torch.cuda.set_device(device)
#             model.to(device)

#             model = nn.parallel.DistributedDataParallel(model,
#                                                         device_ids=[device])
#         else:
#             model.to(device)
#             model = nn.parallel.DistributedDataParallel(model)
#     elif device.type == 'cuda':
#         model = nn.DataParallel(model, device_ids=None).cuda()

#     return model
