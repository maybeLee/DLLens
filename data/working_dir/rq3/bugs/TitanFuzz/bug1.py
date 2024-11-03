"""
Library version: tensorflow (2.10.0)

Crash happens when calling tf.nn.conv2d_transpose:
F tensorflow/core/kernels/mkl/mkl_conv_grad_input_ops.cc:527] Check failed: TensorShapeUtils::IsVector(input_tensor.shape()) == true (0 vs. 1)
Aborted (core dumped)
"""

import tensorflow as tf
# Please run this script
import os
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_HOME"] = ""
os.environ["LD_LIBRARY_PATH"] = ""

input_tensor = tf.Variable(tf.random.normal([3, 28, 28, 3]))
target_input = tf.random.normal([1, 1, 28, 28])
source_input = tf.keras.utils.get_source_inputs(target_input, input_tensor)
target_input = tf.keras.utils.get_source_inputs(source_input, input_tensor)
deconv = tf.nn.conv2d_transpose(target_input, 128, 4, strides=2, padding='SAME')
loss = tf.keras.losses.categorical_crossentropy(outputs, target_input)