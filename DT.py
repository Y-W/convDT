import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class DT_node:
    def __init__(self, label_dist, train_set):
        self.terminal = True
        self.label_dist = label_dist
        self.train_set = train_set
    def set_splitter(self, split_fn):
        pass

class DT:
    def __init__(self):
        pass