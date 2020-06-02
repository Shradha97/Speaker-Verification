from __future__ import division
import os
import math
import time
import pathlib

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import librosa


# For the trained model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, ReLU
from tensorflow.keras.layers import PReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization


audio_dir = '/Users/apple/Desktop/test/audio_data_SLR22'
tfjs_dir = 'TF-js'
#tflite_model_file = 'converted_model.tflite'


nmfcc = 48 # No. of mel coefficients
nframes = nmfcc # No. of frames per partial utterance

# Setting model variables
n_filters = 411
kernel_dim = 24
stride = kernel_dim

# For restoring model
model_number = 2



print("-----------------------------------------------------------")

# Module to build the model
def create_model(out_speakers = 2):
	# Obtaining units of FC1
	fc1_units = math.pow((nmfcc-kernel_dim)//stride+1, 2)*n_filters

	model = Sequential()

	# Conv 1
	model.add(Conv2D(filters = n_filters, kernel_size = (kernel_dim, kernel_dim), strides = (stride, stride), input_shape = (48, 48, 1)))
	model.add(BatchNormalization())
	model.add(PReLU(shared_axes = [1, 2]))

	# Flatten
	model.add(Flatten())

	# FC 1
	model.add(Dense(units = int(256)))
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(Dropout(0.2))

	# FC 2
	model.add(Dense(units = 256))
	model.add(BatchNormalization())
	model.add(ReLU())

	# FC 3
	model.add(Dense(units = 256))
	model.add(BatchNormalization())
	model.add(ReLU())

	# Adding final output layer to the model
	#model.add(Dropout(0.2))
	model.add(Dense(units = out_speakers, activation = 'softmax'))

	return model




def get_saved_model_info(path):
	loaded = tf.contrib.saved_model.load_keras_model(saved_model_path)
	
	#loaded = tf.keras.models.load_model(export_dir)    # To see the input and output signatures of the saved model
	print(list(loaded.signatures.keys()))
	infer = loaded.signatures["serving_default"]
	print(infer.structured_input_signature)
	print(infer.structured_outputs)




def write_tf_model(tflite_quant_model):
	tflite_file = os.path.join(os.path.join(audio_dir, tf_lite_dir), tflite_model_file)
	print("Saving the converted model to ", tflite_file, '\n')
	# Write the model to a file
	with open(tflite_file, "wb") as f:
		f.write(tflite_quant_model) 



def get_model():
	#tf.compat.v1.disable_eager_execution()
	
	# Reloading the model
	print("Restoring the Trained Model...\n")
	try:
		model = tf.keras.models.load_model(os.path.join(os.path.join(audio_dir, "Saved_Model"), 'Model_{}.h5'.format(model_number)))
		print("\nModel loaded from %s\n"%os.path.join(os.path.join(audio_dir, "Saved_Model"), 'Model_{}.h5'.format(model_number)))
	except ValueError:
		raise ValueError("The required model does not exist! Check model path or model number.\n")
	
	return model



if __name__ == "__main__":
	#os.makedirs(os.path.join(audio_dir, tfjs_dir), exist_ok = True)
	os.makedirs(os.path.join(os.path.join(audio_dir, tfjs_dir), "Saved_Models"), exist_ok = True)
	os.makedirs(os.path.join(os.path.join(audio_dir, tfjs_dir), "JSON_Models"), exist_ok = True)

	export_path = os.path.join(os.path.join(audio_dir, tfjs_dir), "Saved_Models")   # Directory containing the saved model
	tfjs_target_path = os.path.join(os.path.join(audio_dir, tfjs_dir), "JSON_Models")

	basemodel = get_model()		# Get the original model
	print("ORIGINAL MODEL ARCHITECTURE")
	basemodel.summary()
	print('\n')


	model = tf.keras.Sequential(basemodel.layers[:-1])
	print("MODEL ARCHITECTURE TO BE USED FOR INFERENCE")
	model.summary()
	print('\n')


	print("Saving the model in tfjs converter compatible format...\n")
	tf.contrib.saved_model.save_keras_model(model, export_path)
	print("\nSaved to %s\n" %export_path)

	print("\nConverting the model to tfjs format...\n")
	tfjs.converters.save_keras_model(model, tfjs_target_path)
	print("\nFinished conversion.\n")

	
	#get_saved_model_info(saved_model_path) # To get info of the saved model

	"""print("Converting the model to tflite format...")	
				converter = tf.contrib.lite.TFLiteConverter.from_saved_model(saved_model_path)	# The folder in which the .pb file is present
				converter.post_training_quantize = True
				tflite_quant_model = converter.convert()    # Converting to tf-lite
				print("Finished conversion.\n")
						
				write_tf_model(tflite_quant_model)    # Write the model to a file"""


