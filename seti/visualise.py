import numpy as np
import cv2 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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



test_transform = transforms.Compose([
		transforms.Resize((256,256))])

fig, ax = plt.subplots(figsize=(20,20))
x = np.load("/Users/pranoyr/Desktop/02504e8ff8d3f66.npy").astype(float)
x = torch.from_numpy(x).view(3,-1,256)
x = test_transform(x)
print(x.shape)
cv2.imshow('window', x.permute(1,2,0).numpy())
ax.imshow(rearrange(x, 't0 t f -> f (t0 t)'))
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()



