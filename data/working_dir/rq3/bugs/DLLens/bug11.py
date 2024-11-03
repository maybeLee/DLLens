"""
Library version: tensorflow (2.10.0)

tf.linalg.solve fails on float16 dtype, but the documentation claims that it should support https://www.tensorflow.org/api_docs/python/tf/linalg/solve

Error:
Could not find device for node: {{node MatrixSolve}} = MatrixSolve[T=DT_HALF, adjoint=true]
All kernels registered for op MatrixSolve:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_HALF]
  device='GPU'; T in [DT_COMPLEX128]
  device='GPU'; T in [DT_COMPLEX64]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
 [Op:MatrixSolve] name: 
"""
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    np.random.seed(23)
    import tensorflow
    import tensorflow as tf
    def tensorflow_call(matrix,rhs,adjoint=False):
        return tf.linalg.solve(matrix,rhs,adjoint)
    matrix = tf.constant(np.random.randn(0,0,0), dtype='float16')
    rhs = tf.constant(np.random.randn(0,1,0,1), dtype='float16')
    adjoint = True
    out = tensorflow_call(matrix,rhs,adjoint)
except Exception as e:
    print("TF Exception: ", e)