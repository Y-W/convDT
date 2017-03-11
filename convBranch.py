# Yijie Wang (wyijie93@gmail.com)

import numpy as np
import tensorflow as tf
import keras

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('scales', 3, 'Number of scales')
tf.app.flags.DEFINE_string('conv_schema', '3,16;3,16;3,16', 'Convolution layers schema')

tf.app.flags.DEFINE_float('balance_loss', 1.0, 'Weight of balance-split loss')

tf.app.flags.DEFINE_integer('training_iterations', 1000, 'Number of training iterations')
tf.app.flags.DEFINE_float('learning_rate_initial', 1e-1, 'Learning rate')

convSeries = [(int(s), int(c)) for s, c in conv.split(',') for conv in FLAGS.conv_schema.split(';')]

def makeConvBranch(input, k_downsamp=FLAGS.scales, convSeries=convSeries, reuse_variable=False):
    with tf.variable_scope('conv_branch', values=[input], dtype=tf.float32, reuse=reuse_variable) as sc:
        scale_outputs = []
        for k in xrange(k_downsamp):
            with tf.variable_scope('scale_%i' % k, [input]):
                last_output = input
                for i, (s, c) in enumerate(convSeries):
                    with tf.variable_scope('conv_%i' % i, [last_output]):
                        filters = tf.get_variable('filters',
                                                  shape=(
                                                      s, s, last_output.get_shape()[3], c),
                                                  initializer=tf.random_normal_initializer(
                                                      stddev=np.sqrt(2.0 / last_output.get_shape()[3])),
                                                  trainable=True)
                        last_output = tf.nn.conv2d(
                            last_output, filters, strides=(1, 1, 1, 1), padding='VALID')
                        bias = tf.get_variable('bias_%i' % i, shape=(c,),
                                               initializer=tf.zeros_initializer(), trainable=True)
                        last_output = tf.nn.relu(last_output + bias)
                last_output = tf.reduce_mean(last_output, axis=(1, 2))
                scale_outputs.append(last_output)
            if k + 1 < k_downsamp:
                _, h, w, _ = input.get_shape()
                input = tf.image.resize_bilinear(
                    input, (h // 2, w // 2), name=('downsamp_%i' % k))
        all_outputs = tf.concat(scale_outputs, axis=1)
        with tf.variable_scope('dense', [all_outputs]):
            weights = tf.get_variable('weights', shape=(all_outputs.get_shape()[1], 1),
                                      initializer=tf.random_normal_initializer(
                                          stddev=np.sqrt(1.0 / all_outputs.get_shape()[1])),
                                      trainable=True)
            bias = tf.get_variable('bias', shape=(1, 1), initializer=tf.zeros_initializer(), trainable=True)
            preact = tf.squeeze(tf.add(tf.matmul(all_outputs, weights), bias), name='preact')
        output = tf.sigmoid(preact, name='branching')
    return output

def gini_impurity(dist):
    return 1.0 - tf.reduce_sum(tf.multiply(dist, dist))

def branching_loss(branching, label, balance_split_weight=FLAGS.balance_loss):
    split_loss = tf.square(0.5 - tf.reduce_mean(branching)) * balance_split_weight
    dist = tf.reduce_mean(tf.multiply(tf.expand_dims(branching, -1), label), axis=0)
    gini_loss = (1.0 - tf.reduce_mean(branching)) * gini_impurity(tf.reduce_mean(label, axis=0) - dist) \
               + tf.reduce_mean(branching) * gini_impurity(dist)
    return split_loss + gini_loss

# tf.logging.set_verbosity(tf.logging.INFO)
def makeNetFn(tf_graph, input_dim, label_dim, net_scope=None):
    assert input_dim[0] == label_dim[0]
    with tf_graph.as_default():
        with tf.variable_scope(net_scope, default_name='conv_branch_train_eval') as sc:
            data = tf.placeholder(tf.float32, shape=input_dim, name='data')
            label = tf.placeholder(tf.float32, shape=label_dim, name='label')
            net_output = makeConvBranch(data)
            net_loss = branching_loss(net_output, label)
            net_eval = tf.greater(net_output, 0.5)
            
            global_step = tf.train.get_or_create_global_step()


