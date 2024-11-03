"""
Library version: tensorflow (2.10.0)

Crash happens when calling tf.raw_ops.MatrixSolveLs:
F tensorflow/core/framework/tensor_shape.cc:186] Non-OK-status: InitDims(dim_sizes) status: INVALID_ARGUMENT: Encountered overflow when multiplying 4560650809926316272 with 5, result: -1
Aborted (core dumped)
"""
import tensorflow as tf
import numpy as np
try:
    with tf.device("CPU"):
        fast = True
        matrix = tf.complex(tf.random.uniform([9, 0, 4560650809926316272, 5], dtype=tf.float64, minval=-1024, maxval=1024),tf.random.uniform([9, 0, 4560650809926316272, 5], dtype=tf.float64, minval=-1024, maxval=1024))
        rhs = tf.complex(tf.random.uniform([4, 5], dtype=tf.float64, minval=-1024, maxval=1024),tf.random.uniform([4, 5], dtype=tf.float64, minval=-1024, maxval=1024))
        l2_regularizer = tf.random.uniform([9, 4, 9, 13, 5, 3], dtype=tf.float64, minval=-1024, maxval=1024)
        res = tf.raw_ops.MatrixSolveLs(
            fast=fast,
            matrix=matrix,
            rhs=rhs,
            l2_regularizer=l2_regularizer,
        )
except Exception as e:
    exception_str = str(e)
    ends_with_exception = True