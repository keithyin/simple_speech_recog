from models import *
import tensorflow as tf
from utils import *

LOG_DIR = 'ckpt/3-bi-rnn-lstm-tanh-l2norm/'

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
    res,edit_distance = sess.run(fetch_list, feed_dict)
    print(edit_distance)
    return res.tolist()


def main(_):
    config = ConfigDeltaTest()
    # prepare data
    root_dir = "data"
    _, test_files= split_file_names(root_dir)
    id2cls, cls2id = generating_cls()
    bg = BatchGenerator(config, _, cls2id=cls2id)
    iter_bg = iter(bg)

    # build model
    model = BiRnnModel(config)
    model.inference()
    model.train_op()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # train model
        saver.restore(sess, LOG_DIR+'rnn-model.ckpt')
        for i in range(config.num_iterations):
            features, labels,  seq_length = next(iter_bg)
            batch_data={}
            batch_data['features'] = features
            batch_data['labels'] = labels
            batch_data['seq_length'] = seq_length
            res = one_iteration(model, batch_data=batch_data, step=i)
            for seq, label in zip(res, labels):
                print("predict:->", end='')
                for word in seq:
                    print(id2cls[word], end='')
                print("------ground-truth:->", end='')
                for word in label:
                    print(id2cls[word], end='')
                print()
            print(['*']*30)

if __name__ == '__main__':
    tf.app.run()