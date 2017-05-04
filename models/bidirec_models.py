"""
bidirectional rnn models
"""

import tensorflow as tf
from models.BaseModel import BaseModel
from tensorflow.contrib import rnn
import re
from tensorflow.contrib import layers


class BiRnnModel(BaseModel):
    def __init__(self, config, rnn_type="lstm", activation_func='tanh', train=True):
        super(BiRnnModel, self).__init__(config, rnn_type, activation_func, train)
        self.num_layers = 3

    def inference(self):
        name = None

        with tf.variable_scope(name, default_name='inference'):
            RNN = None
            if self.rnn_type == "lstm":
                RNN = rnn.LSTMCell
            else:
                RNN = rnn.GRUCell
            output = stacked_bidirectional_rnn(RNN, self.config.hidden_size, self.num_layers,
                                      self.inputs, self.seq_lengths, self.config.batch_size)
            # rnn_cell_fw = rnn.MultiRNNCell([rnn_cell_fw(self.config.hidden_size,
            #                                             activation=self.activation_func) for _ in range(self.num_layers)])
            # rnn_cell_bw = rnn.MultiRNNCell([rnn_cell_bw(self.config.hidden_size,
            #                                             activation=self.activation_func) for _ in range(self.num_layers)])
            #
            # initial_state_fw = rnn_cell_fw.zero_state(self.config.batch_size, dtype=tf.float32)
            # initial_state_bw = rnn_cell_bw.zero_state(self.config.batch_size, dtype=tf.float32)
            #
            # (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, self.inputs, self.seq_lengths,
            #                                                   initial_state_fw, initial_state_bw,dtype=tf.float32)
            #
            # output = tf.concat(output, axis=2)

            weight = tf.get_variable("weights", shape=[self.config.hidden_size*2, self.config.num_classes])

            bias = tf.get_variable("bias", shape=[self.config.num_classes], initializer=tf.zeros_initializer())

            reshaped_output = tf.reshape(output, shape=[-1, self.config.hidden_size*2])
            logits = tf.nn.bias_add(tf.matmul(reshaped_output, weight), bias)
            logits = tf.reshape(logits, shape=[self.config.batch_size, -1, self.config.num_classes])
            self.logits = tf.transpose(logits, perm=(1, 0, 2))

            self.prediction, _ = tf.nn.ctc_greedy_decoder(self.logits, sequence_length=self.seq_lengths)
            self.infer = tf.sparse_tensor_to_dense(self.prediction[0])

    def train_op(self):
        name = None

        with tf.variable_scope(name, default_name='train_op'):
            global_step = tf.Variable(0, trainable=False)
            global_step_add_1 = tf.assign_add(global_step, 1)
            lr_initial = 0.01
            lr_decay = tf.train.exponential_decay(lr_initial, global_step, decay_steps=1000,
                                                  decay_rate=0.999)
            tf.summary.scalar("lr", lr_decay)

            add_to_weights()

            # regularization
            l2_norm = layers.l2_regularizer(0.001)
            regularization = layers.apply_regularization(l2_norm, tf.get_collection(tf.GraphKeys.WEIGHTS))


            if self.logits is None:
                raise ValueError("you must call inference first!")
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.sparse_labels, self.logits,
                                                      sequence_length=self.seq_lengths))+regularization
            tf.summary.scalar("loss", self.loss)
            self.edit_distance = tf.reduce_mean(tf.edit_distance(tf.cast(self.prediction[0], tf.int32),
                                                                 self.sparse_labels))
            tf.summary.scalar("edit_distance", self.edit_distance)
            # lr 0.01 0.002
            with tf.control_dependencies([global_step_add_1]):
                #opt = tf.train.RMSPropOptimizer(0.01, momentum=0.99)
                opt = tf.train.GradientDescentOptimizer(lr_decay)
                gradients = tf.gradients(self.loss, tf.trainable_variables())
                # avoiding gradient exploding
                gradients = [tf.clip_by_value(gradient, -1, 1) for gradient in gradients]
                self.optimizer = opt.apply_gradients(zip(gradients, tf.trainable_variables()))

            with tf.name_scope('gradients_summary'):
                for gradient in gradients:
                    tf.summary.histogram(gradient.name, gradient)
            with tf.name_scope('value_summary'):
                for val in tf.trainable_variables():
                    tf.summary.histogram(val.name, val)
            self.merge_summary = tf.summary.merge_all()


def add_to_weights():
    variables = tf.trainable_variables()
    for variable in variables:
        matched = re.findall(r'weights', variable.name)
        if matched:
            print(variable.op.name)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, variable)


def stacked_bidirectional_rnn(RNN, num_units, num_layers, inputs, seq_lengths, batch_size):
    """
    multi layer bidirectional rnn
    :param RNN: RNN class
    :param num_units: hidden unit of RNN cell
    :param num_layers: the number of layers
    :param inputs: the input sequence
    :param seq_lengths: sequence length
    :param batch_size:
    :return: the output of last layer bidirectional rnn with concatenating
    """
    _inputs = inputs
    for _ in range(num_layers):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = RNN(num_units)
            rnn_cell_bw = RNN(num_units)
            initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                                                              initial_state_fw, initial_state_bw, dtype=tf.float32)
            _inputs = tf.concat(output, 2)
    return _inputs
