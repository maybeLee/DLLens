"""
Library version: tensorflow (2.10.0)
tf.math.erfinv outputs incorrect result: -inf on when input is invalid (-1.2640526647764023).
The expected output is: NaN

We reported a similar issue, and it has been confirmed and fixed by developers.
Related link:
https://github.com/tensorflow/tensorflow/issues/59576
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(30)


import tensorflow
import tensorflow as tf
def tensorflow_call(input):
    return tf.math.erfinv(input)

input = tf.constant(np.random.randn(), dtype='float64')
out = tensorflow_call(input)
print(out)
