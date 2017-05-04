import tensorflow as tf
from tensorflow.contrib import rnn
from models.BaseModel import BaseModel


class Model(object):
    pass


class RnnModel(BaseModel):
    def __init__(self, config, rnn_type="lstm",activation_func='tanh',train=True):
        super(RnnModel, self).__init__(config, rnn_type, activation_func, train)

    def inference(self):
        name = None

        with tf.variable_scope(name, default_name='inference'):
            rnn_cell = None
            if self.rnn_type == "lstm":
                rnn_cell = rnn.LSTMCell
            else:
                rnn_cell = rnn.GRUCell
            rnn_cell = rnn.MultiRNNCell([rnn_cell(self.config.hidden_size, self.activation_func) for _ in range(2)])
            initial_state = rnn_cell.zero_state(self.config.batch_size, dtype=tf.float32)
            (output, state) = tf.nn.dynamic_rnn(rnn_cell, self.inputs, sequence_length=self.seq_lengths,
                                                initial_state=initial_state)
            weight = tf.get_variable("weights", shape=[self.config.hidden_size, self.config.num_classes])
            bias = tf.get_variable("bias", shape=[self.config.num_classes], initializer=tf.zeros_initializer())
            reshaped_output = tf.reshape(output, shape=[-1, self.config.hidden_size])
            logits = tf.nn.bias_add(tf.matmul(reshaped_output, weight), bias)
            logits = tf.reshape(logits, shape=[self.config.batch_size, -1, self.config.num_classes])
            self.logits = tf.transpose(logits, perm=(1, 0, 2))

            self.prediction,_ = tf.nn.ctc_greedy_decoder(self.logits, sequence_length=self.seq_lengths)
            self.infer = tf.sparse_tensor_to_dense(self.prediction[0])

    def train_op(self):
        name = None

        with tf.variable_scope(name, default_name='train_op'):
            global_step = tf.Variable(0, trainable=False)
            global_step_add_1 = tf.assign_add(global_step, 1)
            lr_initial = 0.001
            lr_decay = tf.train.exponential_decay(lr_initial, global_step, decay_steps=100,
                                                  decay_rate=0.9)
            tf.summary.scalar("lr", lr_decay)
            if self.logits is None:
                raise ValueError("you must call inference first!")
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.sparse_labels, self.logits,
                                                      sequence_length=self.seq_lengths))
            tf.summary.scalar("loss", self.loss)
            self.edit_distance = tf.reduce_mean(tf.edit_distance(tf.cast(self.prediction[0], tf.int32),
                                              self.sparse_labels))
            tf.summary.scalar("edit_distance", self.edit_distance)
            # lr 0.01 0.002
            with tf.control_dependencies([global_step_add_1]):
                #RMSprop
                #opt = tf.train.RMSPropOptimizer(0.01, momentum=0.99)
                opt = tf.train.GradientDescentOptimizer(lr_decay)
                gradients = tf.gradients(self.loss, tf.trainable_variables())
                #avoiding gradient exploding
                gradients = [tf.clip_by_value(gradient, -0.1, 0.1) for gradient in gradients]
                self.optimizer = opt.apply_gradients(zip(gradients, tf.trainable_variables()))

            with tf.name_scope('gradients_summary'):
                for gradient in gradients:
                    tf.summary.histogram(gradient.name, gradient)
            with tf.name_scope('value_summary'):
                for val in tf.trainable_variables():
                    tf.summary.histogram(val.name, val)
            self.merge_summary = tf.summary.merge_all()