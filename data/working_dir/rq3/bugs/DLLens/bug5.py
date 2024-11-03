"""
Library version: tensorflow (2.10.0)
tf.math.is_strictly_increasing outputs incorrect result `True` when input is an uint64 tensor.
The expected output is: `False`

We reported a similar issue, and it has been confirmed and fixed by a developer.
Related link:
https://github.com/tensorflow/tensorflow/issues/62072
"""



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(44)


import tensorflow
import tensorflow as tf
def tensorflow_call(x):
  return tf.math.is_strictly_increasing(x)

x = tf.constant(np.random.randint(-10, 10, (2)), dtype='uint64')

out = tensorflow_call(x)
print(out)
