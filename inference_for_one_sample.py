from models import *
import tensorflow as tf
from utils import *
from utils import process_audio

LOG_DIR = 'ckpt/2-bi-rnn-lstm-tanh-l2norm-aug/'


def one_iteration(model, batch_data, step, sess=None):
    features = batch_data['features']
    seq_length = batch_data['seq_length']
    if sess is None:
        sess = tf.get_default_session()
    if sess is None:
        raise ValueError("you must pass a session or using with tf.Session() as sess")

    feed_dict = {model.inputs: features,
                 model.seq_lengths: seq_length}
    fetch_list = [model.infer]
    res, = sess.run(fetch_list, feed_dict)
    return res.tolist()


def process_one_file(filename, cls2id):
    feature = process_audio(filename)
    features = [(feature - np.mean(feature)) / np.std(feature)]  # normalize
    seq_lengths = [len(feature)]
    return np.array(features).astype(np.float32), \
           np.array(seq_lengths).astype(np.int32)


def main(_):
    config = ConfigDeltaTest()
    config.batch_size = 1
    # prepare data
    filename = "data/0001/2709482218.wav"
    # print(len(test_files))
    id2cls, cls2id = generating_cls_for_digit()

    # build model
    model = BiRnnModel(config)
    model.inference()
    model.train_op()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # train model
        saver.restore(sess, LOG_DIR + 'rnn-model.ckpt')

        features, seq_length = process_one_file(filename, cls2id)

        batch_data = {}
        batch_data['features'] = features
        batch_data['seq_length'] = seq_length
        res = one_iteration(model, batch_data=batch_data, step=0)

        for seq in res:
            print("predict:->", end='')
            for word in seq:
                print(id2cls[word], end='')
            print("")
        print('*' * 30)


if __name__ == '__main__':
    tf.app.run()
