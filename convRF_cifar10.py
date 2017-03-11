# Yijie Wang (wyijie93@gmail.com)

import numpy as np
import tensorflow as tf
from convBranch import ConvBranch

np.random.seed(43)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('cifar10_dir', None, 'Directory for cifar10 data')

class Cifar10:
    def __init__(self, data_dir=FLAGS.cifar10_dir):
        self.data_dir = data_dir
        self.num_class = 10
        self.img_dim = (32, 32, 3)
        self.loadCifar10()

    @staticmethod
    def loadBatch(fileName):
        with open(fileName, 'rb') as f:
            dict = cPickle.load(f)
            data = dict['data']
            labels = dict['labels']
        return data, np.array(labels, dtype=np.int8)
    
    @staticmethod
    def processDataLabels(data, labels):
        data = (np.reshape(data, (-1, 3, 32, 32)) * (1.0 / 255.0)).astype(np.float32)
        data = np.transpose(data, axes=(0, 2, 3, 1))
        labels = labels.astype(np.int8)
        # labels = np.eye(self.num_class, dtype=np.float32)[labels]
        shuffle = np.random.permutation(data.shape[0])
        return data[shuffle], labels[shuffle]
    
    @staticmethod
    def loadTrainSet():
        data1, labels1 = Cifar10.loadBatch('cifar10/data_batch_1')
        data2, labels2 = Cifar10.loadBatch('cifar10/data_batch_2')
        data3, labels3 = Cifar10.loadBatch('cifar10/data_batch_3')
        data4, labels4 = Cifar10.loadBatch('cifar10/data_batch_4')
        data = np.concatenate((data1, data2, data3, data4), axis=0)
        labels = np.concatenate((labels1, labels2, labels3, labels4), axis=0)
        return Cifar10.processDataLabels(data, labels)

    @staticmethod
    def loadValidSet():
        data, labels = Cifar10.loadBatch('cifar10/data_batch_5')
        return Cifar10.processDataLabels(data, labels)

    @staticmethod
    def loadTestSet():
        data, labels = Cifar10.loadBatch('cifar10/test_batch')
        return Cifar10.processDataLabels(data, labels)

    def loadCifar10(self):
        self.trainData, self.trainLabelRaw = Cifar10.loadTrainSet()
        self.validData, self.validLabelRaw = Cifar10.loadValidSet()
        self.testData, self.testLabelRaw = Cifar10.loadTestSet()

        self.trainLabel = np.eye(self.num_class, dtype=np.float32)[self.trainLabelRaw]
        self.validLabel = np.eye(self.num_class, dtype=np.float32)[self.validLabelRaw]
        self.testLabel = np.eye(self.num_class, dtype=np.float32)[self.testLabelRaw]

        self.pixelMean = np.mean(self.trainData, axis=0)
        self.trainData -= self.pixelMean
        self.validData -= self.pixelMean
        self.testData -= self.pixelMean

        self.pixelVar = np.mean(self.trainData * self.trainData, axis=0)
        self.trainData /= np.sqrt(self.pixelVar)
        self.validData /= np.sqrt(self.pixelVar)
        self.testData /= np.sqrt(self.pixelVar)

def main(argv=None):
    assert FLAGS.model_dir is not None and FLAGS.cifar10_dir is not None

    np.random.seed(43)
    cifar10 = Cifar10()
    batchSize = 1024
    randSelect = np.random.randint(cifar10.trainData.shape[0], size=(batchSize,))
    img_inputs = cifar10.trainData[randSelect]
    img_labels = cifar10.trainLabel[randSelect]

    test = ConvBranch('test', (batchSize,) + cifar10.img_dim, (batchSize, cifar10.num_class))
    test.train(img_inputs, img_labels)

    eval_inputs = [cifar10.trainData[i] for i in xrange(cifar10.trainData.shape[0])]
    eval_outputs = test.eval(eval_inputs)

    from collections import Counter
    c0, c1 = Counter()
    for i in xrange(cifar10.trainData.shape[0]):
        if not eval_outputs[i]:
            c0[cifar10.trainLabelRaw[i]] += 1
        else:
            c1[cifar10.trainLabelRaw[i]] += 1
    for i in xrange(cifar10.num_class):
        print i, c0[0], c1[i]



if __name__ == '__main__':
    tf.app.run()
