import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms
import numpy as np
from timm.models import convit_small
from timm.models import create_model



model= convit_small(img_size=256, num_classes=1, pretrained=False)
# model = create_model(
#         "convit_small",
#         pretrained=False,
#         num_classes=1,
#         drop_rate=0,
#         drop_path_rate=0.1,
#         drop_block_rate=None,
#         local_up_to_layer=10,
#         locality_strength=1,
#         embed_dim = 48,
#     )
print(model)

x = torch.Tensor(1,3,256,256)
x = model(x)
print(x.shape)

from timm.optim import AdamW
optimizer = AdamW()


# train_transform = A.Compose([
# 	A.Resize(256,256),
# 	A.HorizontalFlip(p=0.5),
# 	A.VerticalFlip(p=0.5),
# 	A.Transpose(),
# 	A.ShiftScaleRotate(),	
# 	A.RandomRotate90(),
# 	ToTensorV2(p=1.0)])



# x = np.ones((546,256,3)) * 2
# x = train_transform(image=x.astype('float32'))['image']
# print(x.shape)
