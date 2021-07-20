

import torch

a = torch.ones(4,2,3)
b = torch.zeros(4,1,3)

print(a)

# print(torch.mm(a,b))
print("****")
print((a*b).shape)
