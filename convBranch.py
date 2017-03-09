import numpy as np
import tensorflow as tf
import keras


def makeConvBranch(input, k_downsamp=3, convSeries=[(3, 16)] * 3):
    with tf.variable_scope('conv_branch', values=[input], dtype=tf.float32) as sc:
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

def branching_loss(branching, label, balance_split_weight=1.0):
    split_loss = tf.square(0.5 - tf.reduce_mean(branching)) * balance_split_weight
    dist = tf.reduce_mean(tf.multiply(tf.expand_dims(branching, -1), label), axis=0)
    gini_loss = (1.0 - tf.reduce_mean(branching)) * gini_impurity(tf.reduce_mean(label, axis=0) - dist) \
               + tf.reduce_mean(branching) * gini_impurity(dist)
    return split_loss + gini_loss

