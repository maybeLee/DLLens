"""
Library version: tensorflow (2.10.0)
tf.math.is_strictly_increasing outputs incorrect result False.
The expected output is: True

We reported a similar issue, and it has been confirmed by a developer.
Related link:
https://github.com/tensorflow/tensorflow/issues/77863
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(89)


import tensorflow
import tensorflow as tf
def tensorflow_call(x):
  return tf.math.is_strictly_increasing(x)
x = tf.constant(np.random.randn(1,4,3), dtype='float64')
out = tensorflow_call(x)
print(out)
