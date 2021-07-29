import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms
import numpy as np
from timm.models import convit_small
from timm.models import create_model
from eff import EfficientNet



model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1, in_channels=1)
x = torch.Tensor(1,1,512,512)
x = model(x)
print(x.shape)

# model= convit_small(img_size=256, num_classes=1, pretrained=False)
# # model = create_model(
# #         "convit_small",
# #         pretrained=False,
# #         num_classes=1,
# #         drop_rate=0,
# #         drop_path_rate=0.1,
# #         drop_block_rate=None,
# #         local_up_to_layer=10,
# #         locality_strength=1,
# #         embed_dim = 48,
# #     )
# print(model)

# x = torch.Tensor(1,3,256,256)
# x = model(x)
# print(x.shape)

# from timm.optim import AdamW
# optimizer = AdamW()


