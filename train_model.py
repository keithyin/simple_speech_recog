"""
1. ckpt/1-bi-rnn-gru-tanh/
2. ckpt/1-bi-rnn-gru-tanh-l2norm/
3. ckpt/1-bi-rnn-lstm-tanh-l2norm/
4. ckpt/2-bi-rnn-lstm-tanh-l2norm/
5. ckpt/2-bi-rnn-lstm-tanh-l1-l2norm/
6. ckpt/2-bi-rnn-lstm-tanh-l2norm-aug/
"""
import tensorflow as tf
from models import *
from digit_utils import *

FLAG = False
COUNTER = 0

LOG_DIR = "ckpt/2-bi-rnn-lstm-tanh-l2norm-aug/"


def one_iteration(model, batch_data, step, writer, sess=None):
    features = batch_data['features']
    labels = batch_data['labels']
    seq_length = batch_data['seq_length']
    if sess is None:
        sess = tf.get_default_session()
    if sess is None:
        raise ValueError("you must pass a session or using with tf.Session() as sess")
    sparse_labels = sparse_tuple_from(labels)

    feed_dict = {model.inputs: features,
                 model.labels: labels,
                 model.sparse_labels: sparse_labels,
                 model.seq_lengths: seq_length}
    fetch_list = [model.optimizer, model.merge_summary, model.edit_distance]
    _, summary, edit_distance = sess.run(fetch_list, feed_dict)
    writer.add_summary(summary=summary, global_step=step)
    print(edit_distance)

    global COUNTER
    global FLAG

    if edit_distance <= 0.01:

        FLAG = True
        COUNTER += 1
        print("COUNTER->", COUNTER)
        if COUNTER >= 10:
            exit()
    else:

        if FLAG:
            COUNTER -= 1
            FLAG = False
            print("COUNTER->", COUNTER)


def main(_):
    tf.set_random_seed(2017)
    config = ConfigDelta()
    # prepare data
    root_dir = "data"
    train_files, _ = split_file_names(root_dir, validate_rate=0)
    # id2cls, cls2id = generating_cls()
    bg = BatchGenerator(config, train_files)

    # build model

    model = BiRnnModel(config, rnn_type='lstm')
    model.inference()
    model.train_op()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # train model
        tf.global_variables_initializer().run()
        # 8180
        # saver.restore(sess, save_path=LOG_DIR+'rnn-model.ckpt')

        writer = tf.summary.FileWriter(logdir=LOG_DIR, graph=sess.graph)
        for i, (features, labels, seq_length) in enumerate(bg):

            batch_data = {}
            batch_data['features'] = features
            batch_data['labels'] = labels
            batch_data['seq_length'] = seq_length
            one_iteration(model, batch_data=batch_data, step=i, writer=writer)
            if i % 10 == 0:
                print("iteration count: ", i)
                saver.save(sess, save_path=LOG_DIR + 'rnn-model.ckpt')


if __name__ == '__main__':
    tf.app.run()
