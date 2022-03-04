import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def parse_single_data(feature, label):
    #define the dictionary -- the structure -- of our single example
    data = {
        'feature' : _bytes_feature(tf.io.serialize_tensor(feature).numpy()),
        'label' : _bytes_feature(tf.io.serialize_tensor(label).numpy())
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def write_data_to_tfr_short(datas, labels, filename:str="data"):
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    for index in range(len(datas)):

        #get the data we want to write
        current_data = datas[index]
        current_label = labels[index]

        out = parse_single_data(feature=current_data, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


if __name__ == '__main__':
    X = np.loadtxt('data/xtrain_copy.txt')
    y = np.loadtxt('data/ytrain_copy.txt')[..., np.newaxis]

    # scaled
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    pickle.dump(sc, open(f'scaler.pkl', 'wb'))

    # random shuffle and split
    seed = 777
    idx = np.arange(X_scaled.shape[0])
    random.seed(seed)
    random.shuffle(idx)
    nb_test_samples = int(0.2 * idx.shape[0])
    X_scaled = X_scaled[idx]
    y = y[idx]
    X_train, y_train = X_scaled[nb_test_samples:], y[nb_test_samples:]
    X_val, y_val = X_scaled[:nb_test_samples], y[:nb_test_samples]

    # write to tf dataset
    write_data_to_tfr_short(X_val, y_val, filename='dataset/val_dataset')
    write_data_to_tfr_short(X_train, y_train, filename='dataset/train_dataset')
