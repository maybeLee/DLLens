"""
Library version: tensorflow (2.10.0)

Crash happens when calling tf.raw_ops.Polygamma:
F tensorflow/core/framework/tensor_shape.cc:186] Non-OK-status: InitDims(dim_sizes) status: INVALID_ARGUMENT: Expected shape dimensions to be non-negative, got -1635801451
Aborted (core dumped)
"""

import tensorflow as tf
import numpy as np
try:
    with tf.device("CPU"):
        a = tf.random.uniform([0, 12, 6, 2214188950942099093], dtype=tf.float32, minval=-18446744073709551615, maxval=18446744073709551615)
        x = tf.random.uniform([1], dtype=tf.float32, minval=-18446744073709551615, maxval=18446744073709551615)
        res = tf.raw_ops.Polygamma(
            a=a,
            x=x,
        )
except Exception as e:
    exception_str = str(e)
    ends_with_exception = True
    