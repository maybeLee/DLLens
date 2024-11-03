"""
Library version: tensorflow (2.10.0)

tf.raw_ops.TruncateMod fails on half dtype, but the documentation claims that it should support https://www.tensorflow.org/api_docs/python/tf/raw_ops/TruncateMod

Error:
Could not find device for node: {{node TruncateMod}} = TruncateMod[T=DT_HALF]
All kernels registered for op TruncateMod:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BFLOAT16, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_BFLOAT16, DT_HALF]
  device='DEFAULT'; T in [DT_INT32]
  device='GPU'; T in [DT_INT32]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_INT64]
  device='CPU'; T in [DT_INT32]
 [Op:TruncateMod]
"""

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    np.random.seed(61)
    import tensorflow
    import tensorflow as tf
    def tensorflow_call(x,y):
        return tf.raw_ops.TruncateMod(x=x,y=y)
    x = tf.constant(np.random.randn(1), dtype='half')
    y = tf.constant(np.random.randn(1,1), dtype='half')
    out = tensorflow_call(x,y)
    
except Exception as e:
    print("TF Exception: ", e)
