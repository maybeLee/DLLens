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

results = dict()
import tensorflow as tf
import time
try:
  try:
    x_tensor = tf.random.uniform([1, 12, 355, 355], dtype=tf.float16)
    x = tf.identity(x_tensor)
    y_tensor = tf.random.uniform([], dtype=tf.float16)
    y = tf.identity(y_tensor)
    results["res_low"] = tf.raw_ops.TruncateMod(x=x,y=y,)
    t_start = time.time()
    results["res_low"] = tf.raw_ops.TruncateMod(x=x,y=y,)
    t_end = time.time()
    results["time_low"] = t_end - t_start
  except Exception as e:
    results["err_low"] = "Error:"+str(e)
  try:
    x = tf.identity(x_tensor)
    x = tf.cast(x, tf.float32)
    y = tf.identity(y_tensor)
    y = tf.cast(y, tf.float32)
    results["res_high"] = tf.raw_ops.TruncateMod(x=x,y=y,)
    t_start = time.time()
    results["res_high"] = tf.raw_ops.TruncateMod(x=x,y=y,)
    t_end = time.time()
    results["time_high"] = t_end - t_start
  except Exception as e:
    results["err_high"] = "Error:"+str(e)
except Exception as e:
  results["err"] = "Error:"+str(e)

print(results)
