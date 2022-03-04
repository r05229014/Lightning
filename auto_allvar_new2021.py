import numpy as np
from netCDF4 import Dataset
import numpy as np
import pickle
import sys
import os 
import time
from sklearn.preprocessing import StandardScaler
#from keras import Model
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import matplotlib.pyplot as plt
#from keras.utils import multi_gpu_model
# own modile
# from Preprocessing import load_alldata, Preprocessing_DNN
# from config import ModelMGPU


def DNN():
	print("Build model!!")
	model = Sequential()

	model.add(Dense(256, activation = 'relu', kernel_initializer='random_uniform',bias_initializer='zeros', input_shape=(11,)))
	for i in range(3):
		model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
	model.add(Dense(1, activation = 'sigmoid',kernel_initializer='random_uniform',bias_initializer='zeros'))
	model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model

def split_shuffle(X,y, TEST_SPLIT=0.2):
        # shuffle
        indices = np.arange(X.shape[0])
        nb_test_samples = int(TEST_SPLIT * X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        X_train = X[nb_test_samples:]
        X_test = X[0:nb_test_samples]
        y_train = y[nb_test_samples:]
        y_test = y[0:nb_test_samples]

        print('X_train shape is : ', X_train.shape)
        print('X_test shape is : ', X_test.shape)
        print('y_train shape is : ', y_train.shape)
        print('y_test shape is : ', y_test.shape)
        print('\n')

        return X_train, X_test, y_train, y_test




#==================================

# Read in Data from splited data   

#==================================





act = sys.argv[1]
if act == 'train':
    for ii in range(10):
        X_tmp=np.loadtxt('../xtrain_copy.txt')
        y_tmp=np.loadtxt('../ytrain_copy.txt')
        X_train, X_test, y_train, y_test = split_shuffle(X_tmp,y_tmp)
#    X_test=np.loadtxt('../xtest_copy.txt')
#    y_test=np.loadtxt('../ytest_copy.txt')
#    time_loc=np.loadtxt('../test_timeloc_copy.txt')
        model = DNN()
        model_save_path = '../model/CNN/Whole_all_var/'
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
#        filepath = model_save_path + "Test_"+str(ii)+"_20210807-{epoch:03d}-{loss:.3f}-.hdf5"
        filepath = model_save_path + "Test_"+str(ii)+"_20220210-{epoch:03d}-{loss:.3f}-.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')# update from ericc
        earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1) #update from markpipi
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=512,epochs=50, shuffle=True, callbacks=[checkpoint, earlystopper])
        with open('../history/DNN/1101.pkl' ,'wb') as f:
            pickle.dump(history.history, f)
        cost = model.evaluate(X_test, y_test, batch_size=1024)
        print(cost)

elif act == 'test':
    X_test=np.loadtxt('../xtest_copy.txt')
    y_test=np.loadtxt('../ytest_copy.txt')
    time_loc=np.loadtxt('../test_timeloc_copy.txt')


    txt_save_path = './Whole_all_var/'
    fn_dir='/home/C.markpipi/ML_lightning/model/CNN/Whole_all_var/' 
    load_path = sys.argv[2]
    fn_list=np.loadtxt(fn_dir+load_path,dtype=str)

    for ii in range(len(fn_list)):
        fn=fn_dir+fn_list[ii]
        model = load_model(fn)
        cost = model.evaluate(X_test, y_test, batch_size=1024)
        print('RMSE = ',cost)
        y_pre = model.predict(X_test, batch_size=1024)
        if not os.path.exists(txt_save_path):
      	    os.mkdir(txt_save_path)


        file_path= txt_save_path+'Test'+str(ii)+'_ypre.txt'
        np.savetxt(file_path,y_pre[:])
        file_path= txt_save_path+'Test'+str(ii)+'_ytes.txt'
        np.savetxt(file_path,y_test[:])
        file_path= txt_save_path+'Test'+str(ii)+'_xtes.txt'
        np.savetxt(file_path,X_test[:])
        file_path= txt_save_path+'Test'+str(ii)+'_time.txt'
        np.savetxt(file_path,time_loc[:])
        plt.figure(figsize=(4,4))
        plt.title('pre ')
        plt.scatter(y_pre, y_test)
        plt.xlim(0,1)
        plt.ylim(0,1)   
        plt.xlabel('predict')
        plt.ylabel('True')
        plt.grid(True)
        del y_pre

#	plt.savefig('../img/DNN/DNN_.png', dpi=300)
        plt.show()
