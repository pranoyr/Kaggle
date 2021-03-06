import numpy as np
import cv2 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

# x = np.random.random((6,224,223))
# print(x.shape)

# x = np.load("/Users/pranoyr/Desktop/0024012d1431fbc.npy")
# print(x.shape)
# x = x[0, :, :] *-255
# print(x)
# print(x.shape)
# cv2.imshow('window', x)
# cv2.waitKey(0)

# x = np.load("/Users/pranoyr/Desktop/0c748de1dbac617.npy")
# print(x.shape)
# x = x[0, :, :] *-255
# print(x)
# print(x.shape)
# cv2.imshow('window', x)
# cv2.waitKey(0)


# x = np.load("/Users/pranoyr/Downloads/002efdabe4e3e45.npy")
# print(x.shape)
# x = x[0, :, :] *-255
# print(x)
# print(x.shape)
# cv2.imshow('window', x)
# cv2.waitKey(0)



import matplotlib.pyplot as plt
from einops import rearrange, reduce
import torch



# test_transform = transforms.Compose([
# 		transforms.Resize((256,256))])

transform = A.Compose([
	A.Resize(512,512),
	A.OneOf([
	A.HorizontalFlip(p=0.5),
	A.VerticalFlip(p=0.5),

	# A.Transpose(),
	A.ShiftScaleRotate(shift_limit= 0.2, scale_limit= 0.2, border_mode=0,
                rotate_limit= 20, value=0, mask_value=0),
	# A.RandomBrightness(limit=0.6, p=0.5),
	A.RandomResizedCrop(scale = [0.9, 1.0], p=1, height=512, width=512),
	
	# 
	# A.RandomRotate90(),
	A.GridDropout( holes_number_x=10, holes_number_y=10, ratio=0.4)
	# A.GridDropout(num_grid=3, mode=0, rotate=15)
	# A.Normalize(mean=[-5.2037e-06, -1.4643e-04,  9.0275e-05], std = [0.9707, 0.9699, 0.9703], max_pixel_value=1, p=1.0),
#	ToTensorV2(p=1.0)

	]),
	
	])

fig, ax = plt.subplots(figsize=(20,20))



for i in range(50):

	# x = cv2.imread("/Users/pranoyr/Desktop/pr.jpeg")
	# x = transform(image=x)['image']
	# cv2.imshow('window', x)
	# cv2.waitKey(0)


    	
	x = np.load("/Users/pranoyr/Desktop/1/008eb601300b642.npy").astype(float)
	x = torch.from_numpy(x).view(3,-1,256)
	
	# x = torch.from_numpy(x).view(3,-1,256)

	x = x.permute(1,2,0).numpy().astype('float32')
	x = x[::2]
	# x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
	x = transform(image=x)['image']
	# x = x.type(torch.FloatTensor)
	print(x.shape)
	

	# ax.imshow(rearrange(x, 't0 t f -> f (t0 t)'))
	# plt.xlabel('Time')
	# plt.ylabel('Frequency')
	# plt.show()

	cv2.imshow('window', x)
	cv2.waitKey(0)
	# break

	
	


	# ax.imshow(rearrange(x, 't0 t f -> f (t0 t)'))
	# plt.xlabel('Time')
	# plt.ylabel('Frequency')
	# plt.show()
	# break


# x = torch.from_numpy(x).view(3,-1,256)
# x = x.permute(1,2,0).numpy()
# print(x.shape)
