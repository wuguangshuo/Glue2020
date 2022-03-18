# a='喜喜哈哈'
# print(a)
# print(list(a))
#
# a=[[1,2,3],[4,5,6]]
# for k1,k2,k3 in a:
#     print(k1)
#
# a={
#     'q':1,
#     'b':2
# }
# print(len(a))
import numpy as np
# test_array = np.random.rand(3, 2)
# test_vector = np.random.rand(4)
# np.savez_compressed('./a', x=test_array, y=test_vector)
# data=np.load('./a.npz',allow_pickle=True)
# x=data['x']
# y=data['y']
# print(x)
# print(y)
# b=x[()]
# print(b)

# a='11'
# print(list(a))

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
# a=torch.rand(4,3)
# m=torch.tensor([0,2,3])
# print(a)
# b=torch.tensor([1,0,1,1])
# print(b.shape)#3
# c=b.nonzero()
# print(c)
# print(c.shape)#2,1
# print(c.squeeze(1))
# d=a[c.squeeze(1)]
# print(d)
# print(a[m])

# a=torch.rand(3,5)
# b=torch.rand(4,5)
# c=[a,b]
# print(c)
# d=pad_sequence(c,batch_first=True)
# print(d)
#
# f=nn.Dropout(0.2)
# e=f(d)
# print(e)

import torch
from torchcrf import CRF
num_tags = 5  # number of tags is 5
model = CRF(num_tags , batch_first=True)
seq_length = 3  # maximum sequence length in a batch
batch_size = 2  # number of samples in the batch
emissions = torch.randn(batch_size,seq_length, num_tags)
tags = torch.tensor([[0,2,3], [1,4,1]], dtype=torch.long)  #(batch_size, seq_length)
print(model(emissions, tags))
#mask
mask = torch.tensor([[1, 1,1], [1, 1,0]], dtype=torch.uint8)
# mask = torch.tensor([[True, True,True], [True, True,False]], dtype=torch.uint8)
model(emissions, tags, mask=mask)