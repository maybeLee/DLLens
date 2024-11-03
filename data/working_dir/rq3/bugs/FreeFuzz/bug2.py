"""
Library version: tensorflow (2.10.0)

tf.linalg.inv fails on half dtype, but the documentation claims that it should support https://www.tensorflow.org/api_docs/python/tf/linalg/inv

Error:
Could not find device for node: {{node MatrixInverse}} = MatrixInverse[T=DT_HALF, adjoint=false]
All kernels registered for op MatrixInverse:
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
 [Op:MatrixInverse]
"""
results = dict()
import tensorflow as tf
import time
try:
  try:
    input_tensor = tf.random.uniform([4, 1, 1], dtype=tf.float16)
    input = tf.identity(input_tensor)
    adjoint = False
    name = None
    results["res_low"] = tf.linalg.inv(input=input,adjoint=adjoint,name=name,)
    t_start = time.time()
    results["res_low"] = tf.linalg.inv(input=input,adjoint=adjoint,name=name,)
    t_end = time.time()
    results["time_low"] = t_end - t_start
  except Exception as e:
    results["err_low"] = "Error:"+str(e)
  try:
    input = tf.identity(input_tensor)
    input = tf.cast(input, tf.float32)
    results["res_high"] = tf.linalg.inv(input=input,adjoint=adjoint,name=name,)
    t_start = time.time()
    results["res_high"] = tf.linalg.inv(input=input,adjoint=adjoint,name=name,)
    t_end = time.time()
    results["time_high"] = t_end - t_start
  except Exception as e:
    results["err_high"] = "Error:"+str(e)
except Exception as e:
  results["err"] = "Error:"+str(e)

print(results)
