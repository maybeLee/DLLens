"""
Library version: torch (2.1.0)
torch.tensor converts uint NaN to large negative integer, which is inconsistent with TensorFlow.

We reported a similar issue, and it has been confirmed by a developer.
Related link:
https://github.com/pytorch/pytorch/issues/138192

"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(11)


import tensorflow
import tensorflow as tf
def tensorflow_call(x,y):
  return tf.raw_ops.Greater(x=x,y=y)

x = tf.constant(np.random.randint(-10, 10, ()), dtype='uint64')
y = tf.constant(np.random.randint(-10, 10, (1,1,1))*np.nan, dtype='uint64')
out = tensorflow_call(x,y)
print(out)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(11)

import torch
import torch.nn.functional as F
def pytorch_call(x, y):
    return torch.gt(x, y)

x = torch.tensor(np.random.randint(-10, 10, ()), dtype=torch.uint8)
y = torch.tensor(np.random.randint(-10, 10, (1,1,1))*np.nan, dtype=torch.uint8)

out = pytorch_call(x,y)

if type(out).__name__ == "Tensor" and out.dtype == torch.bfloat16:
    out = out.to(torch.float)
print(out)
