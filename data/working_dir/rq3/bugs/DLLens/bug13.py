"""
Library version: tensorflow (2.10.0)

tf.raw_ops.ArgMax fails on complex64 dtype, but the 2.10.0 documentation claims that it should support

Error:
Could not find device for node: {{node ArgMax}} = ArgMax[T=DT_COMPLEX64, Tidx=DT_INT64, output_type=DT_INT64]
All kernels registered for op ArgMax:
  device='XLA_CPU_JIT'; output_type in [DT_INT32, DT_INT16, DT_INT64, DT_UINT16]; Tidx in [DT_INT32, DT_INT16, DT_INT64]; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 16005131165644881776, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]
  device='XLA_GPU_JIT'; output_type in [DT_INT32, DT_INT16, DT_INT64, DT_UINT16]; Tidx in [DT_INT32, DT_INT16, DT_INT64]; T in [DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16, 16005131165644881776, DT_UINT16, DT_COMPLEX128, DT_HALF, DT_UINT32, DT_UINT64]
  device='GPU'; T in [DT_BOOL]; output_type in [DT_INT32]; Tidx in [DT_INT32]
  device='GPU'; T in [DT_BOOL]; output_type in [DT_INT64]; Tidx in [DT_INT32]
  device='GPU'; T in [DT_DOUBLE]; output_type in [DT_INT32]; Tidx in [DT_INT32]
  device='GPU'; T in [DT_DOUBLE]; output_type in [DT_INT64]; Tidx in [DT_INT32]
  device='GPU'; T in [DT_FLOAT]; output_type in [DT_INT32]; Tidx in [DT_INT32]
  device='GPU'; T in [DT_FLOAT]; output_type in [DT_INT64]; Tidx in [DT_INT32]
  device='GPU'; T in [DT_HALF]; output_type in [DT_INT32]; Tidx in [DT_INT32]
...
"""

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    np.random.seed(77)
    import tensorflow
    import tensorflow as tf
    def tensorflow_call(input,dimension,output_type=tf.dtypes.int64):
        return tf.raw_ops.ArgMax(input=input,dimension=dimension,output_type=output_type)
    input = tf.constant(np.random.randn(1,1), dtype='complex64')
    dimension = tf.constant(np.random.randint(-10, 10, (1)), dtype='int64')
    out = tensorflow_call(input,dimension)
except Exception as e:
    print("TF Exception: ", e)
