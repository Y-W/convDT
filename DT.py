import random
import numpy as np
import tensorflow as tf
from convBranch import ConvBranch

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('min_node_sample', 100, 'min_node_sample')
tf.app.flags.DEFINE_float('max_split_bias', 0.4, 'max_split_bias')
tf.app.flags.DEFINE_integer('max_samp_size', 128, 'max_samp_size')
tf.app.flags.DEFINE_integer('max_expand_trail', 5, 'max_expand_trail')


class DT_node(object):
    pass

class DT:
    def __init__(self, tree_name, train_data, train_label):
        self.tree_name = str(tree_name)
        self.train_data = train_data
        self.data_shape = train_data.shape[1:]
        self.train_label = train_label
        self.label_n = np.max(train_label) + 1
        self.nodes = []
        root = DT_node()
        root.is_leaf = True
        root.train_ids = range(train_data.shape[0])
        root.expand_trail_cnt = 0
        root.pred = self.get_pred(root.train_ids)
        root.pred_soft = self.get_pred_soft(root.train_ids)
        root.id = len(self.nodes)
        self.nodes.append(root)

        self.expand_tree()

    def get_data(self, ids):
        return self.train_data[ids], self.train_label[ids]

    def sample_data(self, ids):
        if len(ids) > FLAGS.max_samp_size:
            ids = random.sample(ids, FLAGS.max_samp_size)
        tmp = np.zeros((len(ids),) + self.data_shape, dtype=np.float32)
        tmp2 = np.zeros((len(ids),), dtype=np.int_)
        for i, p in enumerate(ids):
            tmp[i] = self.train_data[p]
            tmp2[i] = self.train_label[p]
        return tmp, tmp2

    def get_label_dist(self, ids):
        dist = np.zeros(self.label_n, dtype=np.int_)
        for p in ids:
            dist[self.train_label[p]] += 1
        return dist

    def get_pred_soft(self, ids):
        dist = self.get_label_dist(ids)
        return dist.astype(np.float32) / np.sum(dist)

    def get_pred(self, ids):
        return np.argmax(self.get_label_dist(ids))

    def should_expand_leaf(self, node):
        if node.expand_trail_cnt >= FLAGS.max_expand_trail:
            return False
        if len(node.train_ids) < FLAGS.min_node_sample:
            return False
        dist = self.get_label_dist(node.train_ids)
        if np.sum(dist) == np.max(dist):
            return False
        return True

    def is_expand_success(self, ids0, ids1):
        ratio = len(ids0) * 1.0 / (len(ids0) + len(ids1))
        split_bias = abs(ratio - 0.5)
        # print 'Node Expansion Test', self.get_label_dist(ids0).tolist(), ratio, self.get_label_dist(ids1).tolist(), 1.0 - ratio
        return split_bias < FLAGS.max_split_bias

    def try_expand_node(self, node):
        conv_br = ConvBranch(self.tree_name + '_br_' + str(node.id), self.data_shape, self.label_n)
        conv_br.train(*self.sample_data(node.train_ids))
        all_data, all_label = self.get_data(node.train_ids)
        split_result = conv_br.eval(all_data)
        ids0 = [p for i, p in enumerate(node.train_ids) if not split_result[i]]
        ids1 = [p for i, p in enumerate(node.train_ids) if split_result[i]]
        if self.is_expand_success(ids0, ids1):
            node.br = conv_br
            node.is_leaf = False
            ln = DT_node()
            rn = DT_node()
            node.left_node = ln
            node.right_node = rn
            
            ln.is_leaf = True
            ln.train_ids = ids0
            ln.expand_trail_cnt = 0
            ln.pred = self.get_pred(ln.train_ids)
            ln.pred_soft = self.get_pred_soft(ln.train_ids)
            ln.id = len(self.nodes)
            self.nodes.append(ln)

            rn.is_leaf = True
            rn.train_ids = ids1
            rn.expand_trail_cnt = 0
            rn.pred = self.get_pred(rn.train_ids)
            rn.pred_soft = self.get_pred_soft(rn.train_ids)
            rn.id = len(self.nodes)
            self.nodes.append(rn)

            return True
        else:
            node.expand_trail_cnt += 1
            return False

    def expand_tree(self):
        cnt_node_pending = 1
        while cnt_node_pending > 0:
            cnt_node_pending = 0
            i = 0
            while i < len(self.nodes):
                n = self.nodes[i]
                if n.is_leaf and self.should_expand_leaf(n) and not self.try_expand_node(n):
                    cnt_node_pending += 1
                i += 1
    
    def eval(self, inputs):
        pending_ids = [None] * len(self.nodes)
        pending_ids[0] = range(inputs.shape[0])
        ans = np.zeros(inputs.shape[0], dtype=np.int_)
        for i in xrange(len(self.nodes)):
            if pending_ids[i] is not None and len(pending_ids[i]) > 0:
                if self.nodes[i].is_leaf:
                    for p in pending_ids[i]:
                        ans[p] = self.nodes[i].pred
                else:
                    node_inputs = inputs[pending_ids[i]]
                    node_results = self.nodes[i].br.eval(node_inputs)
                    ids0 = [p for j, p in enumerate(pending_ids[i]) if not node_results[j]]
                    ids1 = [p for j, p in enumerate(pending_ids[i]) if node_results[j]]
                    pending_ids[self.nodes[i].left_node.id] = ids0
                    pending_ids[self.nodes[i].right_node.id] = ids1
            pending_ids[i] = None
        return ans

    def eval_soft(self, inputs):
        pending_ids = [None] * len(self.nodes)
        pending_ids[0] = range(inputs.shape[0])
        ans = np.zeros((inputs.shape[0], self.label_n), dtype=np.float32)
        for i in xrange(len(self.nodes)):
            if pending_ids[i] is not None and len(pending_ids[i]) > 0:
                if self.nodes[i].is_leaf:
                    for p in pending_ids[i]:
                        ans[p] = self.nodes[i].pred_soft
                else:
                    node_inputs = inputs[pending_ids[i]]
                    node_results = self.nodes[i].br.eval(node_inputs)
                    ids0 = [p for j, p in enumerate(pending_ids[i]) if not node_results[j]]
                    ids1 = [p for j, p in enumerate(pending_ids[i]) if node_results[j]]
                    pending_ids[self.nodes[i].left_node.id] = ids0
                    pending_ids[self.nodes[i].right_node.id] = ids1
            pending_ids[i] = None
        return ans

class RandForest:
    def __init__(self):
        self.trees = []
    def add_tree(self, tree):
        self.trees.append(tree)
    def eval(self, inputs):
        ans = np.zeros((inputs.shape[0], self.trees[0].label_n), dtype=np.float32)
        for t in self.trees:
            ans += t.eval_soft(inputs)
        return np.argmax(ans, axis=1)

def score(truth, pred):
    precision = np.mean(truth == pred)
    label_n = max(np.max(truth), np.max(pred)) + 1
    conf_mtx = np.dot(np.eye(label_n)[truth].T, np.eye(label_n)[pred])
    return precision, conf_mtx
