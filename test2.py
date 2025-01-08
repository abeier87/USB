import torch

W_class_dist = torch.tensor([
    [0.0958], 
    [0.0904], 
    [0.1026], 
    [0.1023], 
    [0.1031], 
    [0.0994], 
    [0.1043], 
    [0.1093], 
    [0.0904], 
    [0.1024]
])
sorted_W_class_dist, sorted_indices = torch.sort(W_class_dist.flatten(), descending=True)
print('sorted_W_class_dist:', sorted_W_class_dist)
print('sorted_indices:', sorted_indices)