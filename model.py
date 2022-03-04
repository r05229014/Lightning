from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout


def DNN():
	model = Sequential()

	model.add(Dense(256, activation = 'relu', kernel_initializer='random_uniform',bias_initializer='zeros', input_shape=(11,)))
	for i in range(3):
		model.add(Dense(512, activation = 'relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
	model.add(Dense(1, activation = 'sigmoid',kernel_initializer='random_uniform',bias_initializer='zeros'))
	model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model
