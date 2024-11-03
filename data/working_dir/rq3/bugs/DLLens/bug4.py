"""
Library version: tensorflow (2.10.0)

tf.keras.activations.sigmoid outputs incorrect result 1.+0.j when input is an inf+complex tensor.
The expected output is: 0.+0.j

It has been fixed in the latest version of TensorFlow.
Related link:
https://github.com/tensorflow/tensorflow/issues/61800
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(62)


import tensorflow
import tensorflow as tf
def tensorflow_call(x):
  return tf.keras.activations.sigmoid(x)

x = tf.constant(np.random.randn(1,1,1)*np.inf, dtype='complex64')

out = tensorflow_call(x)
print(out)
