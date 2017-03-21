import os
import random
import numpy as np
import tensorflow as tf
from DT import DT, RandForest, score
from cifar10 import Cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('tree_input_size', 4096, 'Size of tree training input')

def main(argv=None):
    assert FLAGS.model_dir is not None and FLAGS.cifar10_dir is not None

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    random.seed(43)
    np.random.seed(43)
    cifar10 = Cifar10()
    
    rf = RandForest()
    for i in xrange(100):
        randSelect = np.random.randint(cifar10.trainData.shape[0], size=(FLAGS.tree_input_size,))
        img_inputs = cifar10.trainData[randSelect]
        img_labels = cifar10.trainLabelRaw[randSelect]


        tree = DT('tree_'+str(i), img_inputs, img_labels)
        print 'DT', i, score(cifar10.validLabelRaw, tree.eval(cifar10.validData))
        rf.add_tree(tree)
        print 'RF', i, score(cifar10.validLabelRaw, rf.eval(cifar10.validData))


if __name__ == '__main__':
    tf.app.run()
