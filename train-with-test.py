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
from utils import *

FLAG = False
SUMMARY_FLAG = True
COUNTER = 0
GLOBAL_STEP = 0
TRAIN_SUMM_FLAG = True
TEST_SUMM_FLAG = True

LOG_DIR = "train-test/ckpt/3-bi-rnn-lstm-tanh-l2norm-aug/"


def one_iteration(model, batch_data, writer, is_training=True):
    global GLOBAL_STEP
    global TRAIN_SUMM_FLAG
    global TEST_SUMM_FLAG
    sess = tf.get_default_session()

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
    if is_training:
        fetch_list = [model.optimizer, model.merge_summary, model.edit_distance]
        _, summary, edit_distance = sess.run(fetch_list, feed_dict)
    else:
        fetch_list = [model.merge_summary, model.edit_distance]
        summary, edit_distance = sess.run(fetch_list, feed_dict)

    if TRAIN_SUMM_FLAG:
        if is_training:
            GLOBAL_STEP += 1
        writer.add_summary(summary=summary, global_step=GLOBAL_STEP)
        TRAIN_SUMM_FLAG = False
    elif (not is_training) and TEST_SUMM_FLAG:
        writer.add_summary(summary=summary, global_step=GLOBAL_STEP)
        TEST_SUMM_FLAG = False

    if edit_distance <= 0.01:
        global COUNTER
        global FLAG
        FLAG = True
        COUNTER += 1
        print("COUNTER->", COUNTER)
        if COUNTER >= 10:
            exit()
    else:
        global COUNTER
        global FLAG
        if FLAG:
            COUNTER -= 1
            FLAG = False
            print("COUNTER->", COUNTER)


def one_epoch(model, train_writer, test_writer, train_file_names, test_file_names, id2cls, cls2id):
    global TRAIN_SUMM_FLAG
    global TEST_SUMM_FLAG
    TRAIN_SUMM_FLAG = True
    TEST_SUMM_FLAG = True

    config = ConfigDelta()
    train_bg = BatchGenerator(config, train_file_names, cls2id)
    test_bg = BatchGenerator(config, test_file_names, cls2id)
    iter_train_bg = iter(train_bg)
    iter_test_bg = iter(test_bg)
    ## training stage
    try:
        print("training...")
        while True:
            features, labels, seq_length = next(iter_train_bg)
            batch_data = {}
            batch_data['features'] = features
            batch_data['labels'] = labels
            batch_data['seq_length'] = seq_length
            one_iteration(model, batch_data, train_writer)
    except EOFError as e:
        print("testing...")
        try:
            while True:
                features, labels, seq_length = next(iter_test_bg)
                batch_data = {}
                batch_data['features'] = features
                batch_data['labels'] = labels
                batch_data['seq_length'] = seq_length
                one_iteration(model, batch_data, test_writer, is_training=False)
        except EOFError as e:
            global GLOBAL_STEP
            print("epoch:", GLOBAL_STEP, "Done!")


def main(_):
    tf.set_random_seed(2017)
    config = ConfigDelta()
    # prepare data
    root_dir = "data"
    train_files, test_files = split_file_names(root_dir)
    id2cls, cls2id = generating_cls()
    # build model
    model = BiRnnModel(config, rnn_type='lstm')
    model.inference()
    model.train_op()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # train model
        tf.global_variables_initializer().run()
        #8180
        #saver.restore(sess, save_path=LOG_DIR+'rnn-model.ckpt')

        train_writer = tf.summary.FileWriter(logdir=LOG_DIR+"train/", graph=sess.graph)
        test_writer = tf.summary.FileWriter(logdir=LOG_DIR+"test/", graph=sess.graph)
        for i in range(100000):
            one_epoch(model, train_writer, test_writer, train_files, test_files, id2cls, cls2id)
            if i % 10 == 0:
                saver.save(sess, save_path=LOG_DIR+'rnn-model.ckpt')
if __name__ == '__main__':
    tf.app.run()