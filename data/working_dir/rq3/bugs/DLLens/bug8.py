"""
Library version: torch (2.1.0)
torch.lerp outputs NaN when `end` is inf.
The expected result should be `-inf` instead of NaN.

This bug is already known by developers.
Related link:
https://github.com/pytorch/pytorch/issues/78484
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(94)


import torch
import torch.nn.functional as F
def pytorch_call(input,end,weight):
  return torch.lerp(input,end,weight)

input = torch.tensor(np.random.randn(1), dtype=torch.float16)
end = torch.tensor(np.random.randn(1,1)*np.inf, dtype=torch.float16)
weight = torch.tensor(np.random.randn(1), dtype=torch.float16)

out = pytorch_call(input,end,weight)
print(out)
