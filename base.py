import tensorflow as tf

FlAGS = tf.app.flags.FLAGS
ps_hosts = ["xx.xx.xx.xx: xxxx"]
worker_hosts = ["xx.xx.xx.xx:xxxx", "xx.xx.xx.xx:xxxx"]

cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()

elif FLAGS.job_name == "worker":
    sess = tf.Session()
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        # build_graph()
        step = 0
        while step < FLAGS.total_step:
            sess.run()
