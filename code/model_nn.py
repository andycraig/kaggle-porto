import tensorflow as tf
from tensorflow.contrib import keras

def get_model(extra_layer=False):
	model = keras.models.Sequential()
	model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
			input_shape=(128, 128, 3)))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(keras.layers.Flatten())
	#model.add(keras.layers.Dense(2048, activation='relu'))
	#model.add(keras.layers.Dropout(0.65))
	model.add(keras.layers.Dense(512, activation='relu'))
	model.add(keras.layers.Dropout(0.55))
	if extra_layer:
		model.add(keras.layers.Dense(256, activation='relu'))
		model.add(keras.layers.Dropout(0.55))
	model.add(keras.layers.Dense(1, activation='sigmoid'))
	sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.8, nesterov = True)
	model.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics=['accuracy'])
	print(model.summary())
	return model
