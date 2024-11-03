"""
Library version: tensorflow (2.10.0)

Crash happens when calling tf.raw_ops.SegmentProd:
OP_REQUIRES failed at segment_reduction_ops_impl.h:94 : INVALID_ARGUMENT: segment_ids should be a vector.
"""

import tensorflow as tf
import numpy as np
try:
    with tf.device("GPU:2"):
        data = tf.random.uniform([2, 15, 6, 7, 12], dtype=tf.bfloat16, minval=-1024, maxval=1024)
        segment_ids = tf.saturate_cast(tf.random.uniform([], minval=-1024, maxval=1024, dtype=tf.int64), dtype=tf.int32)
        res = tf.raw_ops.SegmentProd(
            data=data,
            segment_ids=segment_ids,
        )
except Exception as e:
    exception_str = str(e)
    ends_with_exception = True