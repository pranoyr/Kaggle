



import torch 


x = torch.Tensor(6,273,256)
x = x.view(3,-1,256)
print(x.shape)