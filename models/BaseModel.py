import tensorflow as tf

def leaky_relu(x):
    res = tf.maximum(x, 0.1*x)
    res = tf.clip_by_value(res, -6, 6)
    return res


class BaseModel(object):
    def __init__(self, config, rnn_type="lstm",activation_func='tanh',train=True):
        self.config = config
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, None, config.feature_size])
        self.sparse_labels = tf.sparse_placeholder(dtype=tf.int32)
        self.labels = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, 4])
        self.seq_lengths = tf.placeholder(dtype=tf.int32, shape=[config.batch_size])
        self.train = train
        if rnn_type!="lstm" and rnn_type!="gru":
            raise ValueError("rnn_type must be string of'lstm' or 'gru'")
        self.rnn_type = rnn_type
        self.activation_func = None
        if activation_func=='relu':
            self.activation_func = tf.nn.relu6
        elif activation_func == 'tanh':
            self.activation_func = tf.nn.tanh
        elif activation_func == "leaky_relu":
            self.activation_func = leaky_relu
        else:
            raise ValueError("activation_func must be 'tanh' or 'relu' or 'leaky_relu")
        self.optimizer = None
        self.logits = None
        self.prediction = None
        self.loss = None
        self.edit_distance = None
        self.merge_summary = None
        self.infer = None

    def inference(self):
        raise NotImplementedError("you need to override this function")

    def train_op(self):
        raise NotImplementedError("you need to override this function")
