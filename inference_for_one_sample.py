from models import *
import tensorflow as tf
from utils import *
from utils import process_audio

LOG_DIR = 'ckpt/2-bi-rnn-lstm-tanh-l2norm-aug/'


def one_iteration(model, batch_data, step, sess=None):
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
    fetch_list = [model.infer, model.edit_distance]
    res, edit_distance = sess.run(fetch_list, feed_dict)
    print(edit_distance)
    print("res.shape ", res.shape)
    return res.tolist()


def process_one_file(filename, cls2id):
    feature = process_audio(filename)
    features = [(feature - np.mean(feature)) / np.std(feature)]  # normalize
    labels = [car_id_to_index(get_audio_digit(filename), cls2id)]
    seq_lengths = [len(feature)]
    return np.array(features).astype(np.float32), np.array(labels), \
           np.array(seq_lengths).astype(np.int32)


def main(_):
    config = ConfigDeltaTest()
    config.batch_size = 1
    # prepare data
    filename = "data/0001/2709482218.wav"
    # print(len(test_files))
    id2cls, cls2id = generating_cls_for_digit()

    print(id2cls)

    # build model
    model = BiRnnModel(config)
    model.inference()
    model.train_op()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # train model
        saver.restore(sess, LOG_DIR + 'rnn-model.ckpt')

        features, labels, seq_length = process_one_file(filename, cls2id)

        batch_data = {}
        batch_data['features'] = features
        batch_data['labels'] = labels
        batch_data['seq_length'] = seq_length
        res = one_iteration(model, batch_data=batch_data, step=0)

        for seq, label in zip(res, labels):
            print("predict:->", end='')
            for word in seq:
                print(id2cls[word], end='')
            print("------ground-truth:->", end='')
            for word in label:
                print(id2cls[word], end='')
            print()
        print('*' * 30)


if __name__ == '__main__':
    tf.app.run()
