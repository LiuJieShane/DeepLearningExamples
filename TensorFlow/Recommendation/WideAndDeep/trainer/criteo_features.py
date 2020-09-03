import tensorflow as tf
import pandas as pd
import numpy as np

CONTINUOUS_COLUMNS = ['I{}'.format(idx) for idx in range(1, 14)]
CATEGORICAL_COLUMNS = ['C{}'.format(idx) for idx in range(1, 27)]
NV_TRAINING_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

EMBEDDING_DIMENSIONS = {'C{}'.format(idx):64 for idx in range(1,27)}
IDENTITY_NUM_BUCKETS = {
 'C1': 103,
 'C2': 374,
 'C3': 718,
 'C4': 917,
 'C5': 43,
 'C6': 9,
 'C7': 1897,
 'C8': 61,
 'C9': 4,
 'C10': 1376,
 'C11': 1623,
 'C12': 746,
 'C13': 1436,
 'C14': 25,
 'C15': 1310,
 'C16': 814,
 'C17': 10,
 'C18': 856,
 'C19': 313,
 'C20': 4,
 'C21': 775,
 'C22': 9,
 'C23': 15,
 'C24': 857,
 'C25': 38,
 'C26': 619
}

def get_feature_columns(force_subset=None):
    # adding the force_subset as a way to directly pass in column changes for testing/profiling
    deep_columns, wide_columns = [], []

    if force_subset is not None:
        training_columns = force_subset
    else:
        training_columns = NV_TRAINING_COLUMNS

    tf.compat.v1.logging.warn('number of features: {}'.format(len(training_columns)))
    for column_name in training_columns:
        if column_name in IDENTITY_NUM_BUCKETS:
            categorical_column = tf.feature_column.categorical_column_with_identity(
					column_name, 
                    num_buckets=IDENTITY_NUM_BUCKETS[column_name])
            wide_columns.append(categorical_column)
            column = tf.feature_column.embedding_column(
					categorical_column,
					dimension=EMBEDDING_DIMENSIONS[column_name],
					combiner='mean')
            deep_columns.append(column)
        else:
            columns = []
            columns.append(tf.feature_column.numeric_column(column_name, shape=(1,)))
            for column in columns:
                wide_columns.append(column)
                deep_columns.append(column)
    tf.compat.v1.logging.warn('deep columns: {}'.format(len(deep_columns)))
    tf.compat.v1.logging.warn('wide columns: {}'.format(len(wide_columns)))
    tf.compat.v1.logging.warn(
        'wide&deep intersection: {}'.format(len(set(wide_columns).intersection(set(deep_columns)))))
    return wide_columns, deep_columns
