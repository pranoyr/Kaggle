



import torch 


x = torch.Tensor(273,256,6)
x = x.view(-1,256,3)
print(x.shape)