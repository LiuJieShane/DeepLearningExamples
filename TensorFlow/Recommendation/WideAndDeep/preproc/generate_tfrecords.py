import timeit
import os
import tensorflow as tf

CONTINUOUS_COLUMNS = ['I{}'.format(idx) for idx in range(1, 14)]
CATEGORICAL_COLUMNS = ['C{}'.format(idx) for idx in range(1, 27)]
NV_TRAINING_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
COLUMNS = ['label'] + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
DEFAULT_VALUES = [[1]] + [[1.0] for _ in range(13)] + [[1] for _ in range(26)]

def csv2tfrecord(input_filename, output_filename):
    print("\nStart to convert {} to {}\n".format(input_filename, output_filename))
    start_time = timeit.default_timer()
    lines = open(input_filename, "r")
    writer = tf.io.TFRecordWriter(output_filename)
    i = 0
    for line in lines:
        if i==0:
            i+=1
            continue
        data = line.split('\t')
        feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(data[0])]))}
        for i in range(1, 14):
            feature['I{}'.format(i)] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(data[i])]))
        for i in range(1, 27):
            feature['C{}'.format(i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(data[i+13])]))
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString()) 
    writer.close()
    print("Successfully convert {} to {}".format(input_filename,output_filename))
    end_time = timeit.default_timer()
    print("\nThe pretraining process ran for {0} minutes\n".format((end_time - start_time) / 60))

if __name__ == "__main__":
    train_csv = "./dataset/criteo/whole_train.tsv"
    train_tf = "./dataset/criteo/whole_train.tfrecord"
    test_csv = "./dataset/criteo/whole_test.tsv"
    test_tf = "./dataset/criteo/whole_test.tfrecord"
    csv2tfrecord(train_csv, train_tf)
    csv2tfrecord(test_csv, test_tf)