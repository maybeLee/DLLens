"""
Library version: torch (2.1.0)

Crash when large negative indexing on empty tensor:
Segmentation fault (core dumped)

We reported a similar issue, and it has been confirmed by a developer.
Related link:
https://github.com/pytorch/pytorch/issues/115415
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(56)


import torch
import torch.nn.functional as F
def pytorch_call(data, indices, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = segment_ids.max() + 1
    segment_sum = torch.zeros(num_segments, dtype=data.dtype, device=data.device)
    for i in range(len(segment_ids)):
        segment_sum[segment_ids[i]] += data[indices[i]].squeeze()
    return segment_sum

data = torch.tensor(np.random.randn(1,1), dtype=torch.float32)
indices = torch.tensor(np.random.randint(-10, 10, (1)), dtype=torch.int32)
segment_ids = torch.tensor(np.random.randint(-10, 10, (1))*np.nan, dtype=torch.int64)
num_segments = torch.tensor(2, dtype=torch.int32)

out = pytorch_call(data,indices,segment_ids,num_segments)

