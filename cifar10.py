import os
import cPickle
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('cifar10_dir', None, 'Directory for cifar10 data')

class Cifar10:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = FLAGS.cifar10_dir
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
    
    def loadTrainSet(self):
        data1, labels1 = Cifar10.loadBatch(os.path.join(self.data_dir, 'data_batch_1'))
        data2, labels2 = Cifar10.loadBatch(os.path.join(self.data_dir, 'data_batch_2'))
        data3, labels3 = Cifar10.loadBatch(os.path.join(self.data_dir, 'data_batch_3'))
        data4, labels4 = Cifar10.loadBatch(os.path.join(self.data_dir, 'data_batch_4'))
        data = np.concatenate((data1, data2, data3, data4), axis=0)
        labels = np.concatenate((labels1, labels2, labels3, labels4), axis=0)
        return Cifar10.processDataLabels(data, labels)

    def loadValidSet(self):
        data, labels = Cifar10.loadBatch(os.path.join(self.data_dir, 'data_batch_5'))
        return Cifar10.processDataLabels(data, labels)

    def loadTestSet(self):
        data, labels = Cifar10.loadBatch(os.path.join(self.data_dir, 'test_batch'))
        return Cifar10.processDataLabels(data, labels)

    def loadCifar10(self):
        self.trainData, self.trainLabelRaw = self.loadTrainSet()
        self.validData, self.validLabelRaw = self.loadValidSet()
        self.testData, self.testLabelRaw = self.loadTestSet()

        # self.trainLabel = np.eye(self.num_class, dtype=np.float32)[self.trainLabelRaw]
        # self.validLabel = np.eye(self.num_class, dtype=np.float32)[self.validLabelRaw]
        # self.testLabel = np.eye(self.num_class, dtype=np.float32)[self.testLabelRaw]

        self.pixelMean = np.mean(self.trainData, axis=0)
        self.trainData -= self.pixelMean
        self.validData -= self.pixelMean
        self.testData -= self.pixelMean

        self.pixelVar = np.mean(self.trainData * self.trainData, axis=0)
        self.trainData /= np.sqrt(self.pixelVar)
        self.validData /= np.sqrt(self.pixelVar)
        self.testData /= np.sqrt(self.pixelVar)
