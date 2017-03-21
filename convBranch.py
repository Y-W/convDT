# Yijie Wang (wyijie93@gmail.com)

import os
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', None, 'Directory for model checkpoints')

# tf.app.flags.DEFINE_integer('scales', 3, 'Number of scales')
# tf.app.flags.DEFINE_string('conv_schema', '5,4', 'Convolution layers schema')
# tf.app.flags.DEFINE_integer('fcl_size', 32, 'Size of fully connected layer')
tf.app.flags.DEFINE_boolean('hard_gate', False, 'Use hard gating')

tf.app.flags.DEFINE_integer('conv_size', 7, 'Convolution layers schema')

tf.app.flags.DEFINE_float('balance_tolerance', 0.1, 'Tolerance of splitting imbalance')
tf.app.flags.DEFINE_float('balance_loss', 100.0, 'Weight of balance-split loss')

tf.app.flags.DEFINE_integer('training_iterations', 100, 'Number of training iterations')
tf.app.flags.DEFINE_float('learning_rate_initial', 1e0, 'Initial learning rate')
# tf.app.flags.DEFINE_float('learning_rate_final', 1e-3, 'Final learning rate')
tf.app.flags.DEFINE_float('weight_decay', 0, 'Weight L2 loss')
tf.app.flags.DEFINE_float('momentum', 0.5, 'Momentum value')

tf.app.flags.DEFINE_integer('eval_batch_size', 1024, 'Maximum size of eval batch')

class ConvBranch:
    def __init__(self, name, input_dim, label_n, model_dir=None):
        if model_dir is None:
            model_dir = FLAGS.model_dir
        self.name = name
        self.model_dir = model_dir
        self.input_dim = input_dim
        self.label_n = label_n
    
    def makeConvBranch(self, reuse_variable=False):
        input = self.data
        conv_size = FLAGS.conv_size
        hard_gate = FLAGS.hard_gate

        with self.tf_graph.as_default():
            with tf.variable_scope('conv_branch', values=[input], dtype=tf.float32, reuse=reuse_variable) as sc:
                with tf.variable_scope('conv', [input]):
                    filters = tf.get_variable('filters',
                                              shape=(conv_size, conv_size, input.get_shape()[3].value, 1),
                                              initializer=tf.random_normal_initializer(
                                              stddev=np.sqrt(2.0 / input.get_shape()[3].value)),
                                              trainable=True)
                    last_output = tf.nn.conv2d(input, filters, strides=(1, 1, 1, 1), padding='SAME')
                    bias = tf.get_variable('bias', shape=(1,),
                                           initializer=tf.zeros_initializer(), trainable=True)
                    last_output = tf.nn.relu(last_output + bias)
                    last_output = tf.reduce_mean(last_output, axis=(1, 2))
                with tf.variable_scope('dense', [last_output]):
                    bias = tf.get_variable('bias', shape=(1,), initializer=tf.zeros_initializer(), trainable=True)
                    preact = tf.squeeze(tf.add(last_output, bias), name='preact')
                if hard_gate:
                    self.branch = tf.hard_gate(tf.tanh(preact), name='branching')
                else:
                    self.branch = tf.sigmoid(preact, name='branching')
                self.branch_result = tf.greater(self.branch, 0.5, name='branching_boolean')


    def makeConvBranch_large(self, reuse_variable=False):
        input = self.data
        k_downsamp = FLAGS.scales
        convSeries = [(int(conv.split(',')[0]), int(conv.split(',')[1])) for conv in FLAGS.conv_schema.split(';')]
        hard_gate = FLAGS.hard_gate
        fcl_size = FLAGS.fcl_size

        with self.tf_graph.as_default():
            with tf.variable_scope('conv_branch', values=[input], dtype=tf.float32, reuse=reuse_variable) as sc:
                scale_outputs = []
                for k in xrange(k_downsamp):
                    with tf.variable_scope('scale_%i' % k, [input]):
                        last_output = input
                        for i, (s, c) in enumerate(convSeries):
                            with tf.variable_scope('conv_%i' % i, [last_output]):
                                filters = tf.get_variable('filters',
                                                        shape=(
                                                            s, s, last_output.get_shape()[3].value, c),
                                                        initializer=tf.random_normal_initializer(
                                                            stddev=np.sqrt(2.0 / last_output.get_shape()[3].value)),
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
                            input, (h.value // 2, w.value // 2), name=('downsamp_%i' % k))
                all_outputs = tf.concat(scale_outputs, axis=1)
                with tf.variable_scope('dense', [all_outputs]):
                    with tf.variable_scope('fcl_0', [all_outputs]):
                        weights = tf.get_variable('weights', shape=(all_outputs.get_shape()[1].value, fcl_size),
                                            initializer=tf.random_normal_initializer(
                                                stddev=np.sqrt(1.0 / all_outputs.get_shape()[1].value)),
                                            trainable=True)
                        bias = tf.get_variable('bias', shape=(fcl_size,), initializer=tf.zeros_initializer(), trainable=True)
                        fcl_0 = tf.nn.relu(tf.add(tf.matmul(all_outputs, weights), bias), name='fcl_0')
                    with tf.variable_scope('fcl_1', [fcl_0]):
                        weights = tf.get_variable('weights', shape=(fcl_0.get_shape()[1].value, 1),
                                            initializer=tf.random_normal_initializer(
                                                stddev=np.sqrt(1.0 / fcl_0.get_shape()[1].value)),
                                            trainable=True)
                        bias = tf.get_variable('bias', shape=(1,), initializer=tf.zeros_initializer(), trainable=True)
                        preact = tf.squeeze(tf.add(tf.matmul(fcl_0, weights), bias), name='preact')
                if hard_gate:
                    self.branch = tf.hard_gate(tf.tanh(preact), name='branching')
                else:
                    self.branch = tf.sigmoid(preact, name='branching')
                self.branch_result = tf.greater(self.branch, 0.5, name='branching_boolean')

    @staticmethod
    def gini_impurity(dist):
        return 1.0 - tf.reduce_sum(tf.multiply(dist, dist))
    
    @staticmethod
    def oneguess_impurity(dist):
        return 1.0 - tf.reduce_max(dist)

    def makeTrainEndPoints(self):
        branching = self.branch
        label = self.label
        balance_split_weight = FLAGS.balance_loss
        weight_l2 = FLAGS.weight_decay

        with self.tf_graph.as_default():
            split_ratio = tf.reduce_mean(branching)
            imbalance = tf.abs(0.5 - split_ratio)
            imbalance = tf.nn.relu(imbalance - FLAGS.balance_tolerance)
            self.split_loss = tf.square(imbalance)

            dist = tf.reduce_sum(tf.multiply(tf.expand_dims(branching, -1), label), axis=0)
            dist_0 = (tf.reduce_sum(label, axis=0) - dist) / tf.maximum(label.get_shape()[0].value - tf.reduce_sum(branching), 1)
            dist_1 = (dist) / tf.maximum(tf.reduce_sum(branching), 1)

            adjusted_split_ratio = tf.clip_by_value(split_ratio, 0.5 - FLAGS.balance_tolerance, 0.5 + FLAGS.balance_tolerance)

            self.gini_loss_0 = ConvBranch.oneguess_impurity(dist_0)
            self.gini_loss_1 = ConvBranch.oneguess_impurity(dist_1)
            self.gini_loss = (1.0 - adjusted_split_ratio) * self.gini_loss_0 \
                    + adjusted_split_ratio * self.gini_loss_1
            self.weight_loss = None
            for p in tf.trainable_variables():
                if self.weight_loss is None:
                    self.weight_loss = tf.nn.l2_loss(p)
                else:
                    self.weight_loss = tf.add(self.weight_loss, tf.nn.l2_loss(p))
            self.total_loss = self.split_loss * balance_split_weight + self.gini_loss + self.weight_loss * weight_l2
    
    def setup_training(self):
        with self.tf_graph.as_default():
            self.global_step = tf.contrib.framework.get_or_create_global_step() # tf.train.get_or_create_global_step()
            # self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_initial, self.global_step, 
            #                          FLAGS.training_iterations, FLAGS.learning_rate_final / FLAGS.learning_rate_initial, 
            #                          staircase=False, name='learning_rate')
            self.learning_rate = FLAGS.learning_rate_initial
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum)
            self.train_step = self.optimizer.minimize(self.total_loss, global_step=self.global_step, name='optimize_step')

    def train(self, data, label):
        label = np.eye(self.label_n, dtype=np.float32)[label]
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.data = tf.placeholder(tf.float32, shape=data.shape, name='data')
            self.label = tf.placeholder(tf.float32, shape=label.shape, name='label')
        self.makeConvBranch()
        self.makeTrainEndPoints()
        self.setup_training()
        with tf.Session(graph=self.tf_graph) as sess:
            sess.run(tf.global_variables_initializer())
            for _ in xrange(FLAGS.training_iterations):
                step, split_loss, gini_loss_0, gini_loss_1, gini_loss, _ = \
                        sess.run((self.global_step, self.split_loss, self.gini_loss_0, self.gini_loss_1, self.gini_loss, self.train_step),
                                                          feed_dict={self.data : data, self.label : label})
            # print 'Step=%i split_loss=%f gini_loss_0=%f gini_loss_1=%f gini_loss=%f' % (step, split_loss, gini_loss_0, gini_loss_1, gini_loss)
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(sess, os.path.join(self.model_dir, self.name+'.ckpt'))
        self.tf_graph = None

    def eval_batches(self, inputs):
        if len(inputs) == 0:
            return []
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.data = tf.placeholder(tf.float32, shape=inputs[0].shape, name='data')
        self.makeConvBranch()
        results = []
        with tf.Session(graph=self.tf_graph) as sess:
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(sess, os.path.join(self.model_dir, self.name+'.ckpt'))
            for input in inputs:
                result = sess.run(self.branch_result, feed_dict={self.data : input})
                results.append(result)
        self.tf_graph = None
        return results
        
    
    def eval(self, inputs):
        if inputs.shape[0] == 0:
            return np.zeros(0)
        batch_size = min(inputs.shape[0], FLAGS.eval_batch_size)
        input_batches = []
        p = 0
        while p < inputs.shape[0]:
            tmp = np.zeros((batch_size,) + inputs.shape[1:], dtype=np.float32)
            k = min(len(inputs) - p, batch_size)
            tmp[:k] = inputs[p:p+k]
            input_batches.append(tmp)
            p += k
        result_batches = self.eval_batches(input_batches)
        results = np.zeros(inputs.shape[0], dtype=np.bool_)
        p = 0
        for r in result_batches:
            k = min(inputs.shape[0] - p, batch_size)
            results[p:p+k] = r[:k]
            p += k
        return results
