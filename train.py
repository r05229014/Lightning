import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from pathlib import Path
from model import DNN


def _parse_data_function(example_proto):
    data_feature_description = {
        'feature' : tf.io.FixedLenFeature([], tf.string),
        'label' : tf.io.FixedLenFeature([], tf.string)
    }

    # Parse the input tf.train.Example proto using the dictionary above.
    features = tf.io.parse_single_example(example_proto, data_feature_description)
    data = tf.io.parse_tensor(features['feature'], "double") 
    label = tf.io.parse_tensor(features['label'], "double")
    data.set_shape([11,])
    label.set_shape([1,])
    return data, label


def get_dataset(dataset, BATCH_SIZE):
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"

	# set
	EXP_NAME = 'Test1'
	AUTOTUNE = tf.data.AUTOTUNE
	BATCH_SIZE = 1024
	MODEL_SAVE_DIR = f'models/{EXP_NAME}/'
	MODEL_SAVE_PATH = MODEL_SAVE_DIR + "{epoch:03d}-{val_acc:.6f}-.hdf5"
	LOG_SAVE_DIR = f'logs/{EXP_NAME}/'
	Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
	Path(LOG_SAVE_DIR).mkdir(parents=True, exist_ok=True)

	# load tfrec dataset
	train_dataset = tf.data.TFRecordDataset('dataset/train_dataset.tfrecords')
	val_dataset = tf.data.TFRecordDataset('dataset/val_dataset.tfrecords')

	# Create a dictionary describing the features.
	train_dataset = train_dataset.map(_parse_data_function, num_parallel_calls=AUTOTUNE)
	val_dataset = val_dataset.map(_parse_data_function, num_parallel_calls=AUTOTUNE)
	train_dataset = get_dataset(train_dataset, BATCH_SIZE)
	val_dataset = get_dataset(val_dataset, BATCH_SIZE)

	# build model
	model = DNN()
	checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_SAVE_DIR)
	model.fit(train_dataset,
           	  validation_data=val_dataset,
              epochs=1000,
              callbacks=[checkpoint, tensorboard_callback])
