"""
Library version: tensorflow (2.10.0)

Crash happens when calling tf.nn.conv2d:
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

input_data = tf.random.uniform(shape=[2, 2, 2, 2], minval=0, maxval=10, dtype=tf.int32, seed=1968)
bias = tf.random.uniform(shape=[2], minval=0, maxval=10.0, dtype=tf.float32)
input_data = tf.cast(input_data, dtype=tf.float32)
input_data = tf.cast(input_data, tf.float32)
output_data = tf.raw_ops.BiasAdd(value=input_data, bias=bias, data_format='NHWC')
output_data = tf.reshape(output_data, [(- 1), 2, 2])
output_data = tf.reshape(output_data, [(- 1), 2])
output_data = tf.identity(output_data)
output_data = tf.cast(output_data, tf.float32)
output_data = tf.multiply(output_data, 0.1)
output_data = tf.nn.relu6(output_data)
output_data = tf.nn.dropout(output_data, 0.5)
output_data = tf.nn.tanh(output_data)
output_data = tf.nn.relu(output_data)
output_data = tf.square(output_data)
output_data = tf.nn.elu(output_data)
output_data = tf.tanh(output_data)
output_data = tf.nn.sigmoid(output_data)
output_data = tf.math.sigmoid(output_data)
output_data = tf.nn.softmax(output_data)
output_data = tf.nn.conv2d(output_data, tf.transpose(input_data, [0, 2, 1, 3]), strides=[1, 1, 1, 1], padding='SAME')
output_data = tf.nn.bias_add(output_data, bias)
output_data = tf.reshape(output_data, [(- 1), 8])