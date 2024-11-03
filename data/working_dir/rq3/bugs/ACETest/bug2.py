"""
Library version: tensorflow (2.10.0)

Crash happens when calling tf.raw_ops.NextAfter:
F tensorflow/core/framework/tensor_shape.cc:186] Non-OK-status: InitDims(dim_sizes) status: INVALID_ARGUMENT: Expected shape dimensions to be non-negative, got -1287400943
Aborted (core dumped)
"""


import tensorflow as tf
import numpy as np
try:
    with tf.device("CPU"):
        x1 = tf.random.uniform([0, 1762761891043856913, 8, 16, 3], dtype=tf.float64, minval=-1024, maxval=1024)
        x2 = tf.random.uniform([3], dtype=tf.float64, minval=-1024, maxval=1024)
        res = tf.raw_ops.NextAfter(
            x1=x1,
            x2=x2,
        )
except Exception as e:
    exception_str = str(e)
    ends_with_exception = True