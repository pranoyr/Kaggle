import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as transforms



transform = A.Compose([
	A.Resize(256,256),
	A.HorizontalFlip(p=0.5),
	A.VerticalFlip(p=0.5),
	A.Transpose(),
	A.ShiftScaleRotate(),	
	A.RandomRotate90(),
	ToTensorV2(p=1.0)

])


x = torch.ones(532,256,3) * 2
x = transform(image=x.numpy())['image']
print(x.shape)
