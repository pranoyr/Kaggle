import torch

x = [
    [1, 0, 1, 0],  # Input 1
    [0, 2, 0, 2],  # Input 2
    [1, 1, 1, 1]  # Input 3
]
x = torch.tensor(x, dtype=torch.float32)


w_key = [
    [0, 0, 1, 0],
    [1, 1, 0, 1],
    [0, 1, 0, 0],
    [1, 1, 0, 1]
]
w_query = [
    [1, 0, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 0]
]
w_value = [
    [0, 2, 0, 0],
    [0, 3, 0, 0],
    [1, 0, 3, 2],
    [1, 1, 0, 1]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)


# generate k,q,v
keys = torch.mm(x, w_key)
print(keys.shape)




querys = torch.mm(x, w_query)
values = torch.mm(x,  w_value)

attn_scores = torch.mm(querys, keys.T)
print(attn_scores.shape)


from torch.nn.functional import softmax

attn_scores_softmax = softmax(attn_scores, dim=-1)

print(attn_scores_softmax.shape)

weighted_values = torch.mul(values[:,None],  attn_scores_softmax.T[:,:,None])

print(values[:,None].shape)
print(attn_scores_softmax.T[:,:,None].shape)
print(weighted_values.shape)


outputs = weighted_values.sum(dim=0)

print(outputs.shape)