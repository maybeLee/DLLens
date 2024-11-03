"""
Library version: tensorflow (2.10.0)
tf.math.special.bessel_y1 outputs incorrect result: nan on -inf.
The expected output is: -inf

We reported a similar issue, and it has been confirmed by a developer.
Related link:
https://github.com/tensorflow/tensorflow/issues/77864
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(41)


import tensorflow
import tensorflow as tf
def tensorflow_call(x):
  return tf.math.special.bessel_y1(x)

x = tf.constant(np.random.randn(4)*np.inf, dtype='double')

out = tensorflow_call(x)
print(out)
