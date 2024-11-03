"""
Library version: tensorflow (2.10.0)

Crash happens when calling tf.math.sobol_sample:
F tensorflow/core/framework/tensor.cc:733] Check failed: 1 == NumElements() (1 vs. 12)Must have a one element tensor
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

num_samples = 2000
dim = 2
skip = tf.constant([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])
sobol_sample = tf.math.sobol_sample(num_samples, dim, skip)