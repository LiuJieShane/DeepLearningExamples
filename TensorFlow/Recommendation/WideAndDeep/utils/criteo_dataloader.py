import tensorflow as tf
from tensorflow.compat.v1 import logging
import pandas as pd

TRAIN_DATASET_SIZE = 1600000
EVAL_DATASET_SIZE = 400000

CONTINUOUS_COLUMNS = ['I{}'.format(idx) for idx in range(1, 14)]
CATEGORICAL_COLUMNS = ['C{}'.format(idx) for idx in range(1, 27)]
NV_TRAINING_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
COLUMNS = ['label'] + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
DEFAULT_VALUES = [[1]] + [[1.0] for _ in range(13)] + [[1] for _ in range(26)]

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """
    每次调用，从TFRecord文件中读取一个大小为batch_size的batch
    Args:
        data_file: TFRecord文件
        num_epochs: 将TFRecord中的数据重复几遍，如果是None，则永远循环读取不会停止
        shuffle: 是否乱序
        batch_size: batch_size大小

    Returns:
        tensor格式的，一个batch的数据
    """
    def parse_fn(record):
        features = { "label": tf.FixedLenFeature([], tf.int64)}
        for i in range(1, 14):
            features['I{}'.format(i)] = tf.FixedLenFeature([], tf.float32)
        for i in range(14, 40):
            features['C{}'.format(i)] = tf.FixedLenFeature([], tf.int64)
        parsed = tf.parse_single_example(record, features)
        # features
        feature_dict = {}
        for i in range(1, 14):
            feature_dict['I{}'.format(i)] = parsed['I{}'.format(i)]
        for i in range(14, 40):
            feature_dict['C{}'.format(i-13)] = parsed['C{}'.format(i)]
        # label
        label = parsed["label"]
        return feature_dict, label

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TFRecordDataset(data_file).map(parse_fn, num_parallel_calls=5)   # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_epochs*batch_size)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

'''
def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.compat.v1.gfile.Exists(data_file), "{0} not found.".format(data_file)

    def parse_csv(line):
        print("Parsing", data_file)
        columns = tf.io.decode_csv(line, record_defaults = DEFAULT_VALUES,field_delim='\t')
        labels = columns[0]
        features = dict(zip(NV_TRAINING_COLUMNS, columns[1:]))
        return features, labels
    
    dataset = tf.data.TextLineDataset(data_file).skip(1)  # skip the header
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_epochs*batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    return iterator.get_next()
'''  

'''
def input_fn(data_file, num_epochs, shuffle):
    df_data = pd.read_csv(
      tf.compat.v1.gfile.Open(data_file),
      names = COLUMNS,
      skipinitialspace=True,
      engine="python",
	    sep='\t',
      skiprows=1)
    labels = df_data["label"].astype(int)
    features = df_data[NV_TRAINING_COLUMNS]
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=dict(features),
        y=labels,
        batch_size=128,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=5)
'''