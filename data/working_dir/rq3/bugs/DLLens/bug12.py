"""
Library version: tensorflow (2.10.0)

tf.math.tan fails on int16 dtype, but the 2.10.0 documentation claims that it should support.

Error:
Could not find device for node: {{node Tan}} = Tan[T=DT_INT16]
All kernels registered for op Tan:
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_COMPLEX128, DT_HALF]
  device='GPU'; T in [DT_COMPLEX128]
  device='GPU'; T in [DT_COMPLEX64]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_COMPLEX128, DT_HALF]
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_HALF]
 [Op:Tan]
"""
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    np.random.seed(84)
    import tensorflow
    import tensorflow as tf
    def tensorflow_call(x):
        return tf.math.tan(x)
    x = tf.constant(np.random.randint(-10, 10, (1))*np.nan, dtype='int16')
    out = tensorflow_call(x)
except Exception as e:
    print("TF Exception: ", e)