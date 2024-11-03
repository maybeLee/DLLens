"""
Library version: tensorflow (2.10.0)

Crash happens when calling tf.nn.conv1d:
F ./tensorflow/core/util/tensor_format.h:427] Check failed: index >= 0 && index < num_total_dims Invalid index from the dimension: 3, 0, C
Aborted (core dumped)
"""

# Please run this script
import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_HOME"] = ""
os.environ["LD_LIBRARY_PATH"] = ""

import tensorflow as tf
import numpy as np

x = np.random.rand(10, 10, 3)
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.nn.conv1d(x, filters=tf.random.normal([3, 3, 3]), stride=1, padding='SAME')
y = tf.cast(y, tf.float32)
y = tf.transpose(y, perm=[0, 2, 1])
y = tf.reshape(y, [(- 1), 10])
with tf.GradientTape() as t:
    loss = tf.nn.conv1d(y, filters=tf.random.normal([3, 3, 3]), stride=1, padding='SAME')