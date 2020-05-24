from __future__ import division
import os
import shutil
import math
import time
import re
import random

import tensorflow as tf
#print(tf.__version__)
from keras import backend as K
import numpy as np
#import matplotlib.pyplot as plt
import librosa
import pickle

#from preprocess import preprocess_data, save_mfcc_to_path, one_hot_spk_label, get_mfcc, break_audio_frames, break_audio, get_label, read_files, read_file, fft_normalize, 
import preprocess as ppr


# For the training model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, ReLU
from tensorflow.keras.layers import PReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy


print("\n\nIT STARTS HERE!!\n\n")

# Paths to files
audio_dir = '/Users/apple/Desktop/BTP/audio_data_SLR22'
list_path = '/Users/apple/Desktop/BTP/audio_data_SLR22/list'
dev_set_path = '/Users/apple/Desktop/BTP/audio_data_SLR22/list/ubm.txt'
enroll_set_path = '/Users/apple/Desktop/BTP/audio_data_SLR22/list/enroll.txt'
eval_set_path = '/Users/apple/Desktop/BTP/audio_data_SLR22/list/eval.txt'
test_set_path = '/Users/apple/Desktop/BTP/audio_data_SLR2/list/test.txt'

# These 2 data paths for containing preprocessed train and test data
train_data_path = 'train_data'
enroll_data_path = 'enrolled_data'
eval_data_path = 'eval_data'
test_data_path = 'test_data'


# Setting variables
'''Taking smaller number of frames while taking spectrogam because
  even lesser number of spectrogram frames will give more information.'''
#sv_spec_frame = 180   # Max. frame number of utterances of TI_SV(in ms) for spectrogram
#sv_mfcc_utter = 505 # Max. utterance length of TI_SV(in ms) for mfcc
frame_range_low = 39  # Min utterance length in terms of frames
frame_range_high = 48 # Max utterance length in terms of frames
hop_train = 0.01   # Hop size(ms)
window = .025 # Window length(ms)
nfft = 512 # FFT kernel size
nmels = 48 # No. of mels for mel-spectrogram
nmfcc = 48 # No. of mel coefficients
nframes = nmfcc # No. of frames per partial utterance

lr = 0.00001   # Learning rate
lr_decay_step = 10    # Number of epochs after which the lr will reduce
epochs = 3
N = 4   # Number of speakers in a training batch
M = 5   # Number of utterances per speaker for training
M_enroll = 6  # Number of utterances used for enrollment
M_eval = 7    # Number of utterances taken from speaker for evaluation
spk_labels = {} # A dictionary to contain label corresponding to each speaker file - GLOBAL VARIABLE
#spk_count_dev = 0   # To keep the active count of the development speakers labelled in dictionary while preprocessing - GLOBAL VARIABLE
dev_set_speakers = 180

# Setting model variables
n_filters = 411
kernel_dim = 24
stride = kernel_dim

# For restoring model
model_number = 2




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

  # FC 2
  model.add(Dense(units = 256))
  model.add(BatchNormalization())
  model.add(ReLU())
  model.add(Dropout(0.3))

  # FC 3
  model.add(Dense(units = 256))
  model.add(BatchNormalization())
  model.add(ReLU())

  # Adding final output layer to the model
  model.add(Dropout(0.2))
  model.add(Dense(units = out_speakers, activation = 'softmax'))

  return model




def get_pickle(data_path, file_name):
  global one_hot_spk_labels
  file = open(os.path.join(data_path, file_name), 'rb')
  one_hot_spk_labels = pickle.load(file)
  #print("One hot labels: ", one_hot_spk_labels)
  file.close()




# Module to concatenate audio frames
def resize_audio(audio_batch):
  add_ons = (nframes - np.shape(audio_batch)[2])/2

  tmp = np.flip(audio_batch[:, :, :math.ceil(add_ons)], 2)    # Flipping first few frames to attach at the beginning so as to avoid losing continuity in the audio features
  audio_batch = np.concatenate([tmp, audio_batch], axis = 2)

  tmp = np.flip(audio_batch[:, :, np.shape(audio_batch)[2]-math.floor(add_ons):], 2)  # Flipping last few frames to attach at the end so as to avoid losing continuity in the audio features
  audio_batch = np.concatenate([audio_batch, tmp], axis = 2)

  return audio_batch


# Selecting random batch for training
def get_random_batch(session, speaker_num, utter_num, path, shuffle = True, utter_start = 0):
  if session == 'inference':
    true_spk_label = []
    spk_list = os.listdir(path)
  elif session == 'evaluate':
    true_spk_label = []
    path = os.path.join(path, "Enrollment_done")
    spk_list = os.listdir(path)
  else:
    if session == 'train':
      get_pickle(path, "One_hot.pickle")
    else:
      true_spk_label = []
    path = os.path.join(path, "Speaker_test")
    spk_list = os.listdir(path)

  np_file_list = [f for f in spk_list if not f.startswith('.')] # To remove the unnecessary hidden files in the accessed directories
  #print("File list", np_file_list)

  total_speakers = len(np_file_list)

  if session == 'train':
    one_hot_encoding = np.empty((0, total_speakers))   # The y label: one hot encoding for training

  if shuffle:
    selected_files = random.sample(np_file_list, speaker_num) # select random N speakers 
  else:
    selected_files = np_file_list[:speaker_num]
  
  utter_batch = np.empty((0, nmfcc, nframes))
  print("Selected: ", selected_files)

  for file in selected_files:
    label = np.array((0, 0))
    utters = np.load(os.path.join(path, file))
    if shuffle:
      utter_index = np.random.randint(0, utters.shape[0], utter_num)    # Select M utterances per speaker
      utter_batch = np.concatenate([utter_batch, utters[utter_index]], axis = 0) 
    else:
      utter_batch = np.concatenate([utter_batch, utters[utter_start: utter_start+utter_num]], axis = 0)   # shape: [(NM), nmfcc, nframes]

    if session == 'train':
      #print("One hot labels: ", one_hot_spk_labels[file.strip(".npy")][1][:total_speakers])
      label = [one_hot_spk_labels[file.strip(".npy")][1][:total_speakers]]*utter_num
      one_hot_encoding = np.concatenate([one_hot_encoding, label], axis = 0)   # each speakers true one-hot encoding label for each of the corresponding utterance, shape: [(NM), total_speakers]
    else:
      label = [file.strip(".npy")]
      true_spk_label = np.concatenate([true_spk_label, label], axis = 0)
  #print("One hot: ", np.shape(one_hot_encoding))
  #print("One hot: ", one_hot_encoding)
  #print("Utterance batch original: ", np.shape(utter_batch))

  if session == 'train':
    # Random slicing of input batch for training
    frame_slice = np.random.randint(frame_range_low, frame_range_high+1)       # Number of audio frames to be selected in order to simulate variable sized audio
    utter_batch = utter_batch[:, :, :frame_slice]

    utter_batch = resize_audio(utter_batch) # To resize the number of frames to nframes by replicating first and last few frames

    return (utter_batch.reshape(-1, nmfcc, nframes, 1)).astype('float32'), one_hot_encoding.astype('float32')
  else:
    return utter_batch.reshape(-1, nmfcc, nframes, 1), true_spk_label
  


# Module to train the model
def train(audio_dir, data_path):
  tf.reset_default_graph()   # Reset graph
  total_dev_speakers = len(os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speakers")))  # Gives number of target speakers
  

  # Draw computational graph
  #input_data = tf.placeholder(shape = [None, nmfcc, nframes, 1], dtype = tf.float32)  # Input batch size = (time length of the partial utterance x total batch size x n_mfcc)
  labels = tf.placeholder(shape = [None, total_dev_speakers], dtype = tf.float32)   # Labels for training speaker models
  lr_tensor = tf.placeholder(dtype= tf.float32)  # learning rate
  global_step = tf.Variable(0, name = 'global_step', trainable = False)

  # Define Model
  # Input->2D-CNN->Dense Layer->Dense Layer->Dense Layer->Output Layer
  model = create_model(total_dev_speakers)

  # Printing the model summary
  print("The Model Summary")
  model.summary()

  # Instatiating a loss function
  loss_fn = tf.reduce_mean(categorical_crossentropy(labels, model.get_layer("dense_3").output))

  # Using Adam optimizer
  train_step = tf.train.AdamOptimizer(lr_tensor).minimize(loss_fn)

  # Record loss
  loss_summary = tf.summary.scalar("Loss", loss_fn)
  merged = tf.summary.merge_all()

  # For saving the model
  saver = tf.train.Saver()

  # Training session
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   # Initializing all variables

    os.makedirs(os.path.join(audio_dir, "Check_Point"), exist_ok = True)   # Folder to save model
    os.makedirs(os.path.join(audio_dir, "Logs"), exist_ok = True)    # Folder to save log

    writer = tf.summary.FileWriter(os.path.join(audio_dir, "Logs"), sess.graph)
    
    lr_factor = 1   # lr decay factor (1/2 after certain iterations->10 for now)
    iteration = 0   # To keep a count of the number of iterations that have taken place till now
    loss_acc = 0    # to keep a count of the running average of the loss
    global epochs
    for epoch in range(epochs):
      for it in range(int(total_dev_speakers*100/(1*1))):
        input_batch, spk_label = get_random_batch('train', 1, 1, os.path.join(audio_dir, data_path)) # DEFINING N=2 AND M=5!!!!
        _, loss_cur, summary = sess.run([train_step, loss_fn, merged], feed_dict = {'conv2d_input:0': input_batch, labels: spk_label, lr_tensor: lr*lr_factor})

        loss_acc +=loss_cur   # accumulated loss for each of the 10 iterations

        if iteration%10 == 0:
          writer.add_summary(summary, iteration)  # Writing at tensorboard

        iteration = iteration + 1
        print("Loss is: ", loss_cur/100)

      # Recording loss at the end of the epoch
      print("Epoch : %d        Loss: %.4f" % (epoch,loss_acc/100))
      loss_acc = 0    # Resetting accumulated loss

      if (epoch + 1)%lr_decay_step == 0:
        lr_factor /= 2      # lr decay
        print("Learning rate is decayed! Current lr: ", lr*lr_factor)
      if (epoch + 1)%1 == 0:
        save_path = saver.save(sess, os.path.join(audio_dir, "Check_Point/model"), global_step = epoch//1)  # Saving the model after every 10 epochs
        print("Model saved in path: %s\n" % save_path)
        



# Main module
if __name__ == "__main__":
  # Add configuration later
    #DO

  # Training the model
  print("TRAINING SESSION...")
  #Preprocesing the data to obtain mfcc or spectrogram for input to the network
  ppr.preprocess_data(audio_dir, dev_set_path, train_data_path, 'train')
  #train(audio_dir, train_data_path)
      

  # Enrolling the speakers
  #print("Enrollment session")
  #preprocess_data(audio_dir, enroll_set_path, enroll_data_path, 'enroll')