'''
Library version: torch (2.1.0)

Crash happens when calling torch.mul when tensor involves float16+empty :
Segmentation fault (core dumped)

We reported several similar issues, and they have been confirmed by developers as already known bugs.
Related link:
https://github.com/pytorch/pytorch/issues/115066, 
https://github.com/pytorch/pytorch/issues/115068,
https://github.com/pytorch/pytorch/issues/121148
'''


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(24)


import torch
import torch.nn.functional as F
def pytorch_call(inputs):
    return torch.mul(inputs[0], inputs[1])

inputs_obj_1 = torch.tensor(np.random.randn(1), dtype=torch.half)
inputs_obj_2 = torch.tensor(np.random.randint(-10, 10, (2,1,0,2)), dtype=torch.int32)
inputs = [inputs_obj_1,inputs_obj_2]

out = pytorch_call(inputs)

