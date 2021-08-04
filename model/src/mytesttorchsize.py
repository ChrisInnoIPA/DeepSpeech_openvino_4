import torch
from torch import onnx

# #>>> x = torch.tensor([[1], [2], [3]])
# #>>> x.size()
# torch.Size([3, 1])
# #>>> x.expand(3, 4)
# tensor([[ 1,  1,  1,  1],
#  [ 2,  2,  2,  2],
#  [ 3,  3,  3,  3]])
# #>>> x.expand(-1, 4)   # -1 意味着不会改变该维度
# tensor([[ 1,  1,  1,  1],
#  [ 2,  2,  2,  2],
#  [ 3,  3,  3,  3]])

#>>> x = torch.Tensor([[1], [2], [3]])
#>>> x.size()
torch.Size([3, 1])
#>>> x.expand(3, 4)
 #1 1
 #1 1
 #2 2 2 2
 #3 3 3 3
 #[torch.FloatTensor of size 3x4]

print(torch.Size,"")