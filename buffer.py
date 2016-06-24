import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('variable_size', 1024, """Variable size.""")
tf.app.flags.DEFINE_string('node_name', None, """Node name (job)""")
tf.app.flags.DEFINE_integer('task_index', 0, """task index""")
tf.app.flags.DEFINE_bool('enable_trace', False, 'Enable trace')

def get_run_op():
    variable_size = int(FLAGS.variable_size)
    with tf.device("/job:ps/task:0"):
        buff = tf.Variable(tf.ones([1, FLAGS.variable_size]))

    with tf.device("/job:worker/task:0"):
        y = tf.round(buff)

    return y

def main(_):
    cluster = tf.train.ClusterSpec({
       "ps": ["geeker-4.news.cs.nyu.edu:2222"],
       "worker": ["geeker-3.news.cs.nyu.edu:2223"]
        }) 
    server = tf.train.Server(cluster,
                            job_name=FLAGS.node_name, 
                            task_index=FLAGS.task_index)

    buf_op = get_run_op()
    init_op = tf.initialize_all_variables()
    with tf.Session("grpc://geeker-3.news.cs.nyu.edu:2223") as sess:
          sess.run(init_op)
          for _ in range(1000):
             result = sess.run(buf_op)

if __name__ == "__main__":
    tf.app.run()
