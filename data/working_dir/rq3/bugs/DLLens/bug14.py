"""
Library version: tensorflow (2.10.0)

tf.raw_ops.Inv fails on int32 dtype, but the documentation claims that it should support https://www.tensorflow.org/api_docs/python/tf/raw_ops/Inv

Error:
Could not find device for node: {{node Inv}} = Inv[T=DT_INT32]
All kernels registered for op Inv:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_COMPLEX128, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT16, DT_INT8, DT_COMPLEX64, DT_INT64, DT_BFLOAT16, DT_COMPLEX128, DT_HALF]
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_BFLOAT16]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_COMPLEX128]
  device='GPU'; T in [DT_COMPLEX64]
  device='GPU'; T in [DT_INT64]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_HALF]
 [Op:Inv]
"""

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    np.random.seed(59)
    import tensorflow
    import tensorflow as tf
    def tensorflow_call(x):
        return tf.raw_ops.Inv(x=x)
    x = tf.constant(np.random.randint(-10, 10, (1)), dtype='int32')
    out = tensorflow_call(x)
except Exception as e:
    print("TF Exception: ", e)