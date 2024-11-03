"""
Library version: tensorflow (2.10.0)

tf.linalg.expm fails on float16 dtype, but the documentation claims that it should support https://www.tensorflow.org/api_docs/python/tf/linalg/expm

Error:
Could not find device for node: {{node MatrixSolve}} = MatrixSolve[T=DT_HALF, adjoint=false]
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
 [Op:MatrixSolve]
"""

results = dict()
import tensorflow as tf
import time
try:
  try:
    input_tensor = tf.random.uniform([1], dtype=tf.float16)
    input = tf.identity(input_tensor)
    name = None
    results["res_low"] = tf.linalg.expm(input=input,name=name,)
    t_start = time.time()
    results["res_low"] = tf.linalg.expm(input=input,name=name,)
    t_end = time.time()
    results["time_low"] = t_end - t_start
  except Exception as e:
    results["err_low"] = "Error:"+str(e)
  try:
    input = tf.identity(input_tensor)
    input = tf.cast(input, tf.float32)
    results["res_high"] = tf.linalg.expm(input=input,name=name,)
    t_start = time.time()
    results["res_high"] = tf.linalg.expm(input=input,name=name,)
    t_end = time.time()
    results["time_high"] = t_end - t_start
  except Exception as e:
    results["err_high"] = "Error:"+str(e)
except Exception as e:
  results["err"] = "Error:"+str(e)

print(results)
