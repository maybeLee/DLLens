'''
Library version: torch (2.1.0)

torch.fmod, floating point exception on torch.fmod when x is large negative integer and y is -1:
Floating point exception (core dumped)

We reported a similar issue, and it has been confirmed by a developer.
Related link:
https://github.com/pytorch/pytorch/issues/120597
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(97)


import torch
import torch.nn.functional as F
def pytorch_call(x, y):
    result = torch.fmod(x, y)
    result = result + y * (result < 0).int()
    return result

x = torch.tensor(np.random.randint(-10, 10, (1,1,1,1,1))*np.inf, dtype=torch.int32)
y = torch.tensor(np.random.randint(-10, 10, (1,1,1,1)), dtype=torch.int32)
out = pytorch_call(x,y)
