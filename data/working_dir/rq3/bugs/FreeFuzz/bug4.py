"""
Library version: tensorflow (2.10.0)

tf.raw_ops.MatrixTriangularSolve fails on half dtype, but the documentation claims that it should support https://www.tensorflow.org/api_docs/python/tf/raw_ops/MatrixTriangularSolve

Error:
Could not find device for node: {{node MatrixTriangularSolve}} = MatrixTriangularSolve[T=DT_HALF, adjoint=true, lower=true]
All kernels registered for op MatrixTriangularSolve:
  device='XLA_CPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_BFLOAT16, DT_COMPLEX128, DT_HALF]
  device='XLA_GPU_JIT'; T in [DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_BFLOAT16, DT_COMPLEX128, DT_HALF]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_COMPLEX64]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_COMPLEX128]
  device='GPU'; T in [DT_COMPLEX64]
 [Op:MatrixTriangularSolve]

"""

results = dict()
import tensorflow as tf
import time
try:
  try:
    matrix_tensor = tf.random.uniform([3, 3], dtype=tf.float16)
    matrix = tf.identity(matrix_tensor)
    rhs_tensor = tf.random.uniform([3, 1], dtype=tf.float16)
    rhs = tf.identity(rhs_tensor)
    lower = True
    adjoint = True
    name = None
    results["res_low"] = tf.raw_ops.MatrixTriangularSolve(matrix=matrix,rhs=rhs,lower=lower,adjoint=adjoint,name=name,)
    t_start = time.time()
    results["res_low"] = tf.raw_ops.MatrixTriangularSolve(matrix=matrix,rhs=rhs,lower=lower,adjoint=adjoint,name=name,)
    t_end = time.time()
    results["time_low"] = t_end - t_start
  except Exception as e:
    results["err_low"] = "Error:"+str(e)
  try:
    matrix = tf.identity(matrix_tensor)
    matrix = tf.cast(matrix, tf.float32)
    rhs = tf.identity(rhs_tensor)
    rhs = tf.cast(rhs, tf.float32)
    results["res_high"] = tf.raw_ops.MatrixTriangularSolve(matrix=matrix,rhs=rhs,lower=lower,adjoint=adjoint,name=name,)
    t_start = time.time()
    results["res_high"] = tf.raw_ops.MatrixTriangularSolve(matrix=matrix,rhs=rhs,lower=lower,adjoint=adjoint,name=name,)
    t_end = time.time()
    results["time_high"] = t_end - t_start
  except Exception as e:
    results["err_high"] = "Error:"+str(e)
except Exception as e:
  results["err"] = "Error:"+str(e)

print(results)
