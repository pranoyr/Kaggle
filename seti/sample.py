



import torch 


x = torch.Tensor(6,273,256)
x = x.view(3,-1,256)
print(x.shape)


from torchvision.models import resnet50

model = resnet50(num_classes=1)
print(model)