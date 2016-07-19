"""
buffer.py times a simple tf.round(buff) on a ones variable of length variable_size

Author:
Eric Cox
"""
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import time
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('variable_size', 1024, """Variable size.""")
tf.app.flags.DEFINE_integer('batch_size', 10, """benchmark iteration size.""")
tf.app.flags.DEFINE_string('node_name', None, """Node name (job)""")
tf.app.flags.DEFINE_integer('task_index', 0, """task index""")
tf.app.flags.DEFINE_bool('full_trace', 0, 'Enable trace')

def get_run_op():
    variable_size = int(FLAGS.variable_size)
    buff = tf.Variable(tf.ones([1, FLAGS.variable_size]))
    y = tf.round(buff)
    return y

def time_tensorflow_run(sess, init_op, buf_op):
    num_steps_burn_in = 10
    total_duration = 0
    with sess:
        sess.run(init_op)
        for i in range(FLAGS.batch_size + num_steps_burn_in):
            if FLAGS.full_trace == 1 and i == num_steps_burn_in - 1:
                run_options = config_pb2.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = config_pb2.RunMetadata()
            else:
                run_options = None
                run_metadata = None

            start_time = time.time()
            _ = sess.run(buf_op, options=run_options, run_metadata=run_metadata)
            duration = time.time() - start_time
            if FLAGS.full_trace == 1 and i == num_steps_burn_in - 1:
                tl = tf.python.client.timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open('full_trace.ctf', 'w') as f:
                    f.write(ctf)

            if i >= num_steps_burn_in:
                if not i % 10:
                    print ('%s: step %d, duration = %.3f' %
                        (datetime.now(), i - num_steps_burn_in, duration))
                total_duration += duration

    avg_total_duration = total_duration / (FLAGS.batch_size)
    print ("total_duration avg = %.6f" % avg_total_duration)

def main(_):
    buf_op = get_run_op()
    init_op = tf.initialize_all_variables()
    session = tf.Session()
    time_tensorflow_run(session, init_op, buf_op)

if __name__ == "__main__":
    tf.app.run()
