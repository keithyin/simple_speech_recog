# author: keithyin
# date: 2017.04.18
from pydub import AudioSegment
import scipy.io.wavfile as wav
import numpy as np
import os
try:
    from python_speech_features import mfcc
    from python_speech_features import delta
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError

PAD_VALUE=0


class Config(object):
    hidden_size = 100
    feature_size = 13
    batch_size = 50
    num_iterations = 50000
    num_classes = 38


class ConfigTest(object):
    hidden_size = 100
    feature_size = 13
    batch_size = 20
    num_iterations = 10000
    num_classes = 38


class ConfigDelta(object):
    hidden_size = 100
    feature_size = 39
    batch_size = 50
    num_iterations = 10000
    num_classes = 38


class ConfigDeltaTest(object):
    hidden_size = 100
    feature_size = 39
    batch_size = 20
    num_iterations = 10000
    num_classes = 38

def generating_cls():
    """
    generating the id2cls and cls2id dict
    :return: (id2cls, cls2id) a tuple of dicts
    """
    id2cls = {}
    id2cls[0] = '鲁'
    #print('鲁','鲁'.encode())
    for i in range(10):
        id2cls[i+1] = str(i)
    i = 11
    for j in range(26):
        char = chr(ord('A')+j)
        id2cls[i+j] = char
    id2cls[37] = '*'
    cls2id = dict(zip(id2cls.values(), id2cls.keys()))
    #print(id2cls)
    return id2cls, cls2id


def split_file_names(root_dir, validate_rate = 0.1):
    all_file_names = get_all_file_names(root_dir)
    num_validate = int(0.1*len(all_file_names))
    train_files = all_file_names[0:-num_validate]
    test_files = all_file_names[-num_validate:]
    return train_files, test_files

def get_all_file_names(root_dir):
    """
    given root dir, return all the file under the dir
    :param root_dir: root dir
    :return: list of paths of file
    """
    single_level_dirs = os.listdir(root_dir)
    file_names = []
    for single_level_dir in single_level_dirs:
        second_level_dirs = os.listdir(root_dir+'/'+single_level_dir)
        for second_level_dir in second_level_dirs:
            prefix = root_dir+'/'+single_level_dir
            file_names.append(prefix+'/'+second_level_dir)
    return file_names


def get_car_id(file_name):
    """
    extract the car id from file name
    :param file_name:  the path to that file
    :return: string
    """
    *_, raw_car_id = file_name.split('/')
    car_id = raw_car_id.replace('.wav','')
    return car_id.strip()


def process_audio(file_name):
    """
    given the file name of the audio, using mfcc to process audio
    :param file_name: string
    :return: processed audio , shape is [None, 13]
    """
    random_db = np.random.randint(-10, 10, size=[])
    song_ = AudioSegment.from_wav(file_name)
    fs = song_.frame_rate
    song = song_+int(random_db)
    #song = song_
    audio = raw2ndarray(song.raw_data, file_name)

    processed_audio = mfcc(audio, samplerate=fs)
    delta1 = delta(processed_audio, 1)
    delta2 = delta(processed_audio, 2)
    res = np.concatenate((processed_audio, delta1, delta2), axis=1)
    return res


def raw2ndarray(raw_data, file_name):
    data = np.fromstring(raw_data, dtype=np.int16)
    try:
        data = np.reshape(data, [-1, 2])
    except Exception as e:
        data = data[:, np.newaxis].repeat(2, axis=-1)
    return data


def car_id_to_index(car_id, cls2id):
    if not isinstance(car_id, str):
        raise ValueError("car_id must be a list")
    cls = []
    for id in car_id:
        #print("id", id.encode())
        cls.append(cls2id[id])
    #print(cls)
    return cls


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return indices, values, shape


class BatchGenerator(object):
    """
    construct a batch generator to generator the next batch
    """
    def __init__(self, config, file_names, cls2id, max_epoch=10000):
        self.file_names = file_names
        self.num_samples = len(self.file_names)
        self.indices = np.arange(0, self.num_samples)
        self.batch_size = config.batch_size
        self.batch_counter = 0
        self.num_batches = self.num_samples // config.batch_size
        self.cls2id = cls2id
        self.counter_epoch = 0
        self.max_epoch =max_epoch

    def __iter__(self):
        return self

    def __next__(self):
        """
        return the next batch
        :return: sequences, labels, seq_length
        - sequences: [batch_size, time_step, feature_size]
        - labels: [batch_size, 7]
        - seq_length: [batch_size]
        """
        if not self.batch_counter < self.num_batches-1:
            np.random.shuffle(self.indices)
            self.counter_epoch += 1
            self.batch_counter = 0
            if self.counter_epoch > self.max_epoch:
                raise EOFError("you have reached the max epoch.")
        batch_indices = self.indices[self.batch_counter*self.batch_size:(self.batch_counter+1)*self.batch_size]
        batch_features = []
        labels = []
        seq_lengths = []
        for index in batch_indices:
            file_name = self.file_names[index]
            feature = process_audio(file_name)
            feature = (feature - np.mean(feature)) / np.std(feature) # normalize
            label = car_id_to_index(get_car_id(file_name), self.cls2id)
            batch_features.append(feature)
            labels.append(label)
            seq_lengths.append(len(feature))
        max_len = max(seq_lengths)
        padded_batch_features = []
        # padding the all sequence to the max length of the batch
        for feature in batch_features:
            pad_length = max_len - len(feature)
            padded_feature = feature.tolist()+[[PAD_VALUE]*39]*pad_length#***************** be careful
            padded_batch_features.append(padded_feature)
        # for val in padded_batch_features:
        #     print(np.array(val).shape)
        self.batch_counter += 1
        return np.array(padded_batch_features).astype(np.float32), np.array(labels), \
               np.array(seq_lengths).astype(np.int32)


def main():
    root_dir = 'data'
    config = ConfigDelta()
    _, test_files = split_file_names(root_dir)
    id2cls, cls2id = generating_cls()
    bg = BatchGenerator(config, _, cls2id=cls2id)
    iter_bg = iter(bg)
    for i in range(10):
        features, labels, seq_lengths = next(iter_bg)
        print(features.shape)
        #print(labels)



if __name__ == '__main__':
    main()