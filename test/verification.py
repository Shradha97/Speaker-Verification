from __future__ import division
import os
import shutil
import math
import time
import re
import random
import sys

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import librosa
import pickle

import preprocess as ppr
from config import get_config


# For the training model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, Flatten, ReLU
from tensorflow.keras.layers import PReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy


print("\n\nIT STARTS HERE!!\n\n")

"""# Paths to files
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
test_data_path = 'test_data'"""


# Setting variables
'''Taking smaller number of frames while taking spectrogam because
  even lesser number of spectrogram frames will give more information.'''
#sv_spec_frame = 180   # Max. frame number of utterances of TI_SV(in ms) for spectrogram
#sv_mfcc_utter = 505 # Max. utterance length of TI_SV(in ms) for mfcc
"""
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
#dev_set_speakers = 180
num_classes  = 6      # Total number of classes to be classified into-TO BE SET MANUALLY

# Setting model variables
n_filters = 411
kernel_dim = 24
stride = kernel_dim

# For restoring model
model_number = 2"""
one_hot_spk_labels = {} # A dictionary to contain label corresponding to each development speaker file - GLOBAL VARIABLE


cfg = get_config()

# Module to build the model
def create_model(out_speakers = 2):
  # Obtaining units of FC1
  fc1_units = math.pow((cfg.nmfcc-cfg.kernel_dim)//cfg.stride+1, 2)*cfg.n_filters


  model = Sequential()

  # Conv 1
  model.add(Conv2D(filters = cfg.n_filters, kernel_size = (cfg.kernel_dim, cfg.kernel_dim), strides = (cfg.stride, cfg.stride), input_shape = (48, 48, 1)))
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
  add_ons = (cfg.nframes - np.shape(audio_batch)[2])/2

  tmp = np.flip(audio_batch[:, :, :math.ceil(add_ons)], 2)    # Flipping first few frames to attach at the beginning so as to avoid losing continuity in the audio features
  audio_batch = np.concatenate([tmp, audio_batch], axis = 2)

  tmp = np.flip(audio_batch[:, :, np.shape(audio_batch)[2]-math.floor(add_ons):], 2)  # Flipping last few frames to attach at the end so as to avoid losing continuity in the audio features
  audio_batch = np.concatenate([audio_batch, tmp], axis = 2)

  return audio_batch


# Selecting random batch for training
def get_random_batch(session, speaker_num, utter_num, path, shuffle = True, utter_start = 0, speaker_start = 0):
  #global num_classes

  if session == 'inference':
    true_spk_label = []
    spk_list = os.listdir(path)
  else:
    if session == 'train':
      get_pickle(path, "One_hot.pickle")
      #get_pickle(path, "One_hot.pickle")
      #num_classes = len(list(one_hot_spk_labels))
    else:
      true_spk_label = []
    path = os.path.join(path, "Speakers")
    spk_list = os.listdir(path)

  np_file_list = [f for f in spk_list if not f.startswith('.')] # To remove the unnecessary hidden files in the accessed directories
  #print("File list", np_file_list)
  #total_speakers = len(np_file_list)

  if session == 'train':
    one_hot_encoding = np.empty((0, cfg.num_classes))   # The y label: one hot encoding for training

  if shuffle:
    selected_files = random.sample(np_file_list, speaker_num) # select random N speakers 
  else:
    selected_files = np_file_list[speaker_start:speaker_start + speaker_num]
  
  utter_batch = np.empty((0, cfg.nmfcc, cfg.nframes))
  #print("Selected: ", selected_files)

  for file in selected_files:
    label = np.array((0, 0))
    utters = np.load(os.path.join(path, file))

    while utters.shape[0] < utter_num:
      utters = np.concatenate([utters, utters], axis = 0)

    if shuffle:
      utter_index = np.random.randint(0, utters.shape[0], utter_num)    # Select M utterances per speaker
      utter_batch = np.concatenate([utter_batch, utters[utter_index]], axis = 0) 
    else:
      utter_batch = np.concatenate([utter_batch, utters[utter_start: utter_start+utter_num]], axis = 0)   # shape: [(NM), nmfcc, nframes]

    if session == 'train':
      #print("One hot labels: ", one_hot_spk_labels[file.strip(".npy")][1][:total_speakers])
      label = [one_hot_spk_labels[file.strip(".npy")][1]]*utter_num
      one_hot_encoding = np.concatenate([one_hot_encoding, label], axis = 0)   # each speakers true one-hot encoding label for each of the corresponding utterance, shape: [(NM), total_speakers]
    else:
      label = [file.strip(".npy")]
      true_spk_label = np.concatenate([true_spk_label, label], axis = 0)
  #print("One hot: ", np.shape(one_hot_encoding))
  #print("One hot: ", one_hot_encoding)
  #print("Utterance batch original: ", np.shape(utter_batch))

  if session == 'train':
    # Random slicing of input batch for training
    frame_slice = np.random.randint(cfg.frame_range_low, cfg.frame_range_high+1)       # Number of audio frames to be selected in order to simulate variable sized audio
    utter_batch = utter_batch[:, :, :frame_slice]

    utter_batch = resize_audio(utter_batch) # To resize the number of frames to nframes by replicating first and last few frames

    return (utter_batch.reshape(-1, cfg.nmfcc, cfg.nframes, 1)).astype('float32'), one_hot_encoding.astype('float32')
  else:
    return utter_batch.reshape(-1, cfg.nmfcc, cfg.nframes, 1), true_spk_label
  


# Module to train the model
def train(audio_dir, data_path, val_data_path):
  tf.reset_default_graph()   # Reset graph
  #global num_classes

  total_dev_speakers = len(os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speakers")))  # Gives number of target speakers
  total_val_speakers = len([f for f in os.listdir(os.path.join(os.path.join(audio_dir, val_data_path), "Speakers")) if not f.startswith('.')])  # Gives number of validation speakers

  #num_classes = total_dev_speakers

  # Draw computational graph
  #input_data = tf.placeholder(shape = [None, cfg.nmfcc, nframes, 1], dtype = tf.float32)  # Input batch size = (time length of the partial utterance x total batch size x n_mfcc)
  labels = tf.compat.v1.placeholder(shape = [None, cfg.num_classes], dtype = tf.float32)   # Labels for training speaker models
  lr_tensor = tf.compat.v1.placeholder(dtype= tf.float32)  # learning rate
  global_step = tf.Variable(0, name = 'global_step', trainable = False)

  # Define Model
  # Input->2D-CNN->Dense Layer->Dense Layer->Dense Layer->Output Layer
  model = create_model(cfg.num_classes)

  # Printing the model summary
  print("\nTHE MODEL SUMMARY\n")
  model.summary()
  prediction = model.get_layer("dense_3").output

  # Instatiating a loss function
  loss_fn = tf.reduce_mean(categorical_crossentropy(labels, prediction))

  # Using Adam optimizer
  train_step = tf.compat.v1.train.AdamOptimizer(lr_tensor).minimize(loss_fn)

  # Defining metrics for accuracy
  tf_metric, tf_metric_update = tf.compat.v1.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(prediction, 1), name="my_metric")

# Isolate the variables stored behind the scenes by the metric operation
  running_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="my_metric")

  # Define initializer to initialize/reset running variables
  running_vars_initializer = tf.compat.v1.variables_initializer(var_list=running_vars)


  # Record loss
  train_loss_summary = tf.compat.v1.summary.scalar("Training Loss", loss_fn)
  validation_loss_summary = tf.compat.v1.summary.scalar("Validation Loss", loss_fn)
  train_accuracy_summary = tf.compat.v1.summary.scalar("Training Accuracy", tf_metric)
  validation_accuracy_summary = tf.compat.v1.summary.scalar("Validation Accuracy", tf_metric)

  # For saving the model
  saver = tf.compat.v1.train.Saver()

  # Training session
  with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())   # Initializing all variables

    #os.makedirs(os.path.join(audio_dir, "Check_Point"), exist_ok = True)   # Folder to save model
    os.makedirs(os.path.join(audio_dir, "Saved_Model"), exist_ok = True)
    os.makedirs(os.path.join(audio_dir, "Logs"), exist_ok = True)    # Folder to save log

    writer = tf.compat.v1.summary.FileWriter(os.path.join(audio_dir, "Logs"), sess.graph)
    
    lr_factor = 1   # lr decay factor (1/2 after certain iterations->10 for now)
    iteration = 0   # To keep a count of the number of iterations that have taken place till now
    loss_acc = 0    # to keep a count of the running average of the loss
    #global epochs
    for epoch in range(cfg.epochs):
      sess.run(running_vars_initializer)
      print("Epoch %d" %epoch)
      for it in range(int(total_dev_speakers*100/(2*cfg.M))):
        input_batch, spk_label = get_random_batch('train', 2, cfg.M, os.path.join(audio_dir, data_path)) # DEFINING N=2 AND M=5!!!!
        _, loss_cur, pred_label, loss_summary = sess.run([train_step, loss_fn, prediction, train_loss_summary], feed_dict = {'conv2d_input:0': input_batch, labels: spk_label, lr_tensor: cfg.lr*lr_factor})

        sess.run(tf_metric_update, feed_dict={'conv2d_input:0': input_batch, labels: spk_label})

        loss_acc +=loss_cur   # accumulated loss for each of the 10 iterations

        if iteration%10 == 0:
          writer.add_summary(loss_summary, iteration)  # Writing at tensorboard

          tr_accuracy, tr_accuracy_summary = sess.run([tf_metric, train_accuracy_summary])
          writer.add_summary(tr_accuracy_summary, iteration)  # Writing at tensorboard
          print("Iteration : %d        Train Loss: %.4f        Train accuracy: %.4f" %(iteration, loss_acc/100, tr_accuracy))

        iteration = iteration + 1

      # Doing the validation after each epoch
      val_input_batch, val_spk_label = get_random_batch('train', total_val_speakers, cfg.M, os.path.join(audio_dir, val_data_path), False)
      val_loss, pred_val_label, val_loss_summary = sess.run([loss_fn, prediction, validation_loss_summary], feed_dict = {'conv2d_input:0': val_input_batch, labels: val_spk_label})
      writer.add_summary(val_loss_summary, epoch)

      # Obtaining the validation accuracy after each epoch
      sess.run(tf_metric_update, feed_dict={'conv2d_input:0': val_input_batch, labels: val_spk_label})
      val_accuracy, val_accuracy_summary = sess.run([tf_metric, validation_accuracy_summary])
      writer.add_summary(val_accuracy_summary, epoch)  # Writing at tensorboard

      # Recording validation loss at the end of the epoch
      print("------------------------------------------------------------------------------")
      print("Epoch : %d        Validation Loss: %.4f        Validation Accuracy: %.4f" %(epoch, val_loss/100, val_accuracy))
      print("------------------------------------------------------------------------------")

      loss_acc = 0    # Resetting accumulated training loss after an epoch

      if (epoch + 1)%cfg.lr_decay_step == 0:
        lr_factor /= 2      # lr decay
        print("Learning rate is decayed! Current lr: ", cfg.lr*lr_factor)
      if (epoch + 1)%1 == 0:
        model.save(os.path.join(os.path.join(audio_dir, "Saved_Model"), "Model_{}.h5".format(epoch)))
        print("Model saved in path: %s\n" %os.path.join(os.path.join(audio_dir, "Saved_Model"), "Model_{}.h5".format(epoch)))
        




# Module to normalize the input        
def normalize(inp):
  return inp/tf.sqrt(tf.reduce_sum(tf.math.square(inp), axis = -1, keepdims = True)+1e-6)




# Module to obtain the averaged embedding over a few considered utterances
def save_av_embedding(embeddings, n, m, speakers, data_path):
  ''' Saving the averaged enrollment utterances into a pickle file.'''

  file_path = os.path.join(data_path, "Speaker_embeddings.pickle")
  data = {} # The dictionary to store 
  #print("Shape: ", np.shape(embeddings))


  for count, spk in enumerate(speakers):
    embedding = normalize(tf.reduce_mean(embeddings[count], axis = 0))  # Obtaining the normalized element wise average of different utterance embeddings
    

    with tf.compat.v1.Session() as sess:
      embed = sess.run(embedding)
      print("Saving %s embeddings data to %s" %(spk, file_path))

      #print("Data now: ", data)

      if os.path.exists(file_path) and os.path.getsize(file_path) > 0: 
        # Non-empty file exists
        #print("Speaker1: ", spk)
        file = open(file_path, 'rb')
        data = pickle.load(file)
        data[spk] = embed     # Adding new data to the already existing data
        #print("Data: ", data)
        file = open(file_path, 'wb')
        pickle.dump(data, file)
      else:
        #print("Speaker2: ", spk)
        data[spk] = embed
        #print("Data: ", data)
        file = open(file_path, 'wb')
        pickle.dump(data, file)




# Module to enroll the speakers
def enroll(audio_dir, data_path, session):
  os.makedirs(os.path.join(os.path.join(audio_dir, data_path), "Enrollment_done"), exist_ok = True)  # To store the preocessed speakers who have been enrolled into the model
  spk_list = [f for f in os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speakers")) if not f.startswith('.')]   # To remove unnecessary hidden files

  speakers_to_enroll = len(spk_list) # Gives number of unenrolled processed speakers present inside the Speaker folder

  #tf.reset_default_graph()
  
  # Redefining the same model so that the old weights can be loaded into it
    # Reloading the model
  print("Restoring the Trained Model...")
  try:
    model = tf.keras.models.load_model(os.path.join(os.path.join(audio_dir, "Saved_Model"), 'Model_{}.h5'.format(cfg.model_number)))
    print("Model loaded from %s\n"%os.path.join(os.path.join(audio_dir, "Saved_Model"), 'Model_{}.h5'.format(cfg.model_number)))
  except ValueError:
    raise ValueError("The required model does not exist! Check model path or model number.\n")
  
  # Routine to get the speaker embeddings 
  embed = normalize(model.get_layer("re_lu_2").output)   # the embedding shape: [batch_size, no. of output units in the FC3 after activation]
  #saver = tf.compat.v1.train.Saver(var_list = tf.global_variables())
  
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    input_batch, spk_label = get_random_batch(session, speakers_to_enroll, cfg.M_enroll, os.path.join(audio_dir, data_path), False)
    #print("Input batch: ", np.shape(input_batch))
    #print("Labels: ", np.shape(spk_label))

    # Copying the enrolled speakers into the enrollment_done folder now...while actual running move into the folder instead of copying into it
    for f in os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speakers")):      #FOR COPYING
      if not f.startswith('.'):
        print("Saving %s "%f.strip(".npy")+"input data to %s"%os.path.join(os.path.join(audio_dir, data_path), "Enrollment_done"))
        shutil.move(os.path.join(os.path.join(os.path.join(audio_dir, data_path), "Speakers"), f), os.path.join(os.path.join(audio_dir, data_path), "Enrollment_done"))


    print("\nENROLLING THE SPEAKER(S)...")
    spk_embeddings = sess.run(embed, feed_dict = {'conv2d_input:0': input_batch})   # Getting the speaker embedding
    #print("Embeddings: ", np.shape(spk_embeddings))

    #spk_embeddings = spk_embeddings.reshape([speakers_to_enroll, M_enroll, -1])   # Reshaping to be of the shape: [Number to enrolled speakers, number of utterances per speaker, number of embedding features]
    #print("Reshaped: ", np.shape(spk_embeddings))
    save_av_embedding(sess.run(tf.reshape(spk_embeddings, shape = [speakers_to_enroll, cfg.M_enroll, -1])), speakers_to_enroll, cfg.M_enroll, spk_label, os.path.join(audio_dir, data_path))

    
    print("\nEnrollment completed.\n")
    """
    for f in os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speakers")):      #FOR MOVING
      shutil.move(f, os.path.join(os.path.join(os.path.join(audio_dir, data_path), "Speaker_embeddings"), "Enrollment_done"))"""






# Module to get the claimed speaker embedding
def get_spk_embedding(file_path):
  if os.path.exists(file_path) and os.path.getsize(file_path) > 0: 
    # Non-empty file exists
    file = open(file_path, 'rb') 
    data = pickle.load(file)
    #print("Data is: ", data)
  else:
    raise ValueError('No enrollments found! Enroll speakers first or check for the enrollments file.\n')
  
  return data





# Module to get the similarity between test utterance and the claimed speaker utterance
def get_similarity(enrolled, test, normalized = True):
  if normalized:
    return tf.reduce_sum(tf.multiply(enrolled, test))
  else:
    x_norm = tf.sqrt(tf.reduce_sum(tf.math.square(enrolled)) + 1e-6)
    y_norm = tf.sqrt(tf.reduce_sum(tf.math.square(test)) + 1e-6)

    return tf.reduce_sum(tf.multiply(enrolled, test))/x_norm/y_norm






# Module to get the similarity matrix - giving similarities between different utterances
def get_similarity_matrix(test_spk, speaker_num, utter_num, file_path):
  enrolled_tensor = tf.compat.v1.placeholder(shape = [None], dtype = tf.float32)
  test_tensor = tf.compat.v1.placeholder(shape = [None], dtype = tf.float32)

  sim = get_similarity(enrolled_tensor, test_tensor)

  #print("Test speaker shape: ", np.shape(test_spk))
  enrolled_speakers = get_spk_embedding(file_path)  # To get embeddings of all the speakers for testing
  S = np.empty((speaker_num, utter_num, len(enrolled_speakers.keys())))

  #print("Enrollment speaker: ", enrolled_speakers.keys())

  S = np.empty((utter_num, len(enrolled_speakers.keys())))


  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(speaker_num):
      for j in range(utter_num):
        for k, enr_spk in enumerate(enrolled_speakers.keys()):
          #print("Enrolled shape: ", np.shape(enrolled_speakers[enr_spk]))
          #print("Test shape: ", np.shape(test_spk[i, j, :]))
          #print("Similarity: ", sess.run(sim, feed_dict = {enrolled_tensor: enrolled_speakers[enr_spk], test_tensor: test_spk[i, j, :]}))
          S[j, k] = sess.run(sim, feed_dict = {enrolled_tensor: enrolled_speakers[enr_spk], test_tensor: test_spk[i, j, :]})
  #print("Matrix shape: ", np.shape(S))
  #print("Matrix: ", S)
    
  return S





# Module to save the threshold values for verification during inference in a pickle file
def save_verification_info(file_path, EER, EER_thres, EER_FAR, EER_FRR):
  data = {}
  data['EER'] = EER
  data['EER_thres'] = EER_thres
  data['EER_FAR'] = EER_FAR
  data['EER_FRR'] = EER_FRR

  file = open(file_path, 'wb')
  pickle.dump(data, file)





# Module to evaluate the performance of the model
def evaluate(audio_dir, data_path, enroll_data_path):
  #tf.reset.default_graph()
  spk_list = [f for f in os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speakers")) if not f.startswith('.')]   # To remove unnecessary hidden files
  spks_to_evaluate = len(spk_list) # Gives number of speakers to be evaluated

  #eval_embed = tf.compat.v1.placeholder(shape = [spks_to_evaluate, M_eval, None], dtype = tf.float32) # The batch size will be 1(Number of speakers)*7(Number of utterances per speaker)
  #spk_embed = tf.compat.v1.placeholder(shape = [spks_to_evaluate*M_eval, None], dtype = tf.float32)

  # Reloading the model
  print("Restoring the Trained Model...")
  try:
    model = tf.keras.models.load_model(os.path.join(os.path.join(audio_dir, "Saved_Model"), 'Model_{}.h5'.format(cfg.model_number)))
    print("Model loaded from %s\n"%os.path.join(os.path.join(audio_dir, "Saved_Model"), 'Model_{}.h5'.format(cfg.model_number)))
  except ValueError:
    raise ValueError("The required model does not exist! Check model path or model number.\n")

  # Routine to get the test speaker embeddings 
  test_embed = normalize(model.get_layer("re_lu_2").output)   # The embedding shape: [batch_size, no. of output units in the FC3]
  

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    time1 = time.time() # for check inference time

    
    for_eer = {}  # To store dynamic FAR and FRR values per threshold value

    for thres in [0.01*i+0.5 for i in range(50)]:
      FA_num = 0; FR_num = 0
      for_eer[thres] = [FA_num, FR_num]


    for spk_index in range(spks_to_evaluate):
      input_batch, spk_label = get_random_batch('evaluate', 1, cfg.M_eval, os.path.join(os.path.join(audio_dir, data_path)), False, 0, spk_index)
      #print("Input batch: ", np.shape(input_batch))
      #print("Labels: ", np.shape(spk_label), '\n')

      spk_embedding = sess.run(tf.reshape(test_embed, shape = [1, cfg.M_eval, -1]), feed_dict = {'conv2d_input:0': input_batch})   # Getting the test speaker embedding
      #print("Shape of test speaker embeddings: ", np.shape(spk_embedding))  
      #print("Speakers are: ", spk_label, '\n')

      S = get_similarity_matrix(spk_embedding, 1, cfg.M_eval, os.path.join(os.path.join(audio_dir, enroll_data_path), "Speaker_embeddings.pickle"))    # Getting the similarity score matrix
      print("Similarity matrix for %s" %spk_label[0])
      print(S)    # Similarity matrix
      print('\n')

      
      # Calculating FAR and FRR
      for thres in [0.01*i+0.5 for i in range(50)]:
        S_thres = S > thres
        #print("Threshold is: ", thres)
        #print("S_thres is: ", S_thres)
 
        FA_num = for_eer[thres][0] + np.sum(S_thres)-np.sum(S_thres[:, spk_index])
        #print("FA_num: ", FA_num)
        FR_num = for_eer[thres][1] + cfg.M_eval - np.sum(S_thres[:, spk_index])
        #print("FR_num: ", FR_num, '\n')

        for_eer[thres] = [FA_num, FR_num]


    # Calculating EER
    diff = sys.maxsize; EER=0; EER_thres = 0; EER_FAR=0; EER_FRR=0

    for thres in list(for_eer.keys()):
      FAR = for_eer[thres][0]/(spks_to_evaluate-1)/cfg.M_eval/spks_to_evaluate
      #print("FAR: ", FAR)
      FRR = for_eer[thres][1]/cfg.M_eval/spks_to_evaluate
      #print("FRR: ", FRR, '\n')

      
      # keeping EER = min difference between FAR and FRR
      if diff > abs(FAR-FRR):
        diff = abs(FAR-FRR)
        EER = (FAR+FRR)/2
        EER_thres = thres
        EER_FAR = FAR
        EER_FRR = FRR

    print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thres,EER_FAR,EER_FRR))
    print("Saving the threshold values to ", os.path.join(audio_dir, "Verification_info.pickle"), '\n')
    save_verification_info(os.path.join(audio_dir, "Verification_info.pickle"), EER, EER_thres, EER_FAR, EER_FRR)




# Module to reload the verification thresholds
def get_verification_info(file_path):
  if os.path.exists(file_path) and os.path.getsize(file_path) > 0: 
    # Non-empty file exists
    file = open(file_path, 'rb') 
    data = pickle.load(file)
    #print("Data is: ", data)
  else:
    raise ValueError('No enrollments found! Save thresholds first or check for the Verification_info file.\n')
  
  return data




# Module to infer the speaker identity from the speaker input
def inference(audio_dir, enroll_path, test_path):
  #tf.reset_default_graph()
  spk_list = [f for f in os.listdir(os.path.join(audio_dir, test_path)) if not f.startswith('.')]   # To remove unnecessary hidden files
  spks_to_infer = len(spk_list) # Gives number of speakers to be evaluated
  #print("speakers: ", spks_to_infer)

  test = tf.compat.v1.placeholder(shape = [spks_to_infer*1, cfg.nmfcc, None, 1], dtype = tf.float32)  # The batch size will be 1(Number of speakers)*1(Number of utterances per speaker) for testing
  spk_embed = tf.compat.v1.placeholder(shape = [None], dtype = tf.float32)       
  enrolled_embed =  tf.compat.v1.placeholder(shape = [None], dtype = tf.float32) # Will take in speaker's enrollment embedding as the input 

  # Reloading the model
  print("Restoring the Trained Model...")
  try:
    model = tf.keras.models.load_model(os.path.join(os.path.join(audio_dir, "Saved_Model"), 'Model_{}.h5'.format(cfg.model_number)))
    print("Model loaded from %s\n"%os.path.join(os.path.join(audio_dir, "Saved_Model"), 'Model_{}.h5'.format(cfg.model_number)))
  except ValueError:
    raise ValueError("The required model does not exist! Check model path or model number.\n")

  
 # Routine to get the test speaker embeddings 
  test_embed = normalize(model.get_layer("re_lu_2").output)   # The embedding shape: [batch_size, no. of output units in the FC3]
  similarity = get_similarity(enrolled_embed, spk_embed)      # Routine to calculate cosine similarity
  #saver = tf.compat.v1.train.Saver(var_list = tf.global_variables())
  enrolled_embed_path = os.path.join(os.path.join(audio_dir, enroll_path), "Speaker_embeddings.pickle")

  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Return a single similarity score after verification
    # Also measuring the inference time taken
    start_time = time.time()  # Beginning of inference

    input_data, spk_label = get_random_batch('inference', spks_to_infer, 1, os.path.join(audio_dir, test_path), False)

    spk_embedding = sess.run(tf.reshape(test_embed, shape = [spks_to_infer, -1]), feed_dict = {'conv2d_input:0': input_data})   # Getting the test speaker embedding
    #print("Shape of test speaker embedding: ", np.shape(spk_embedding))  

    enrolled_spk_embed_data = get_spk_embedding(enrolled_embed_path)    # Getting the enrollment speaker embeddings
    #print("Enrollment data: ",  enrolled_spk_embed_data)

    enrolled_spk_embed = np.empty((0, np.shape(spk_embedding)[1]))
    for index, label in enumerate(spk_label):
      if label.split('_')[0] not in enrolled_spk_embed_data.keys():
        print("%s is an invalid identity"%label.split('_')[0])
      else:
        tmp1 = enrolled_spk_embed_data[label.split('_')[0]]
        tmp2 = spk_embedding[index]
        sim_score = sess.run(similarity, feed_dict = {enrolled_embed : tmp1, spk_embed : tmp2})    # Getting the similarity score
        print("Score: ", sim_score)

        thresholds = get_verification_info(os.path.join(audio_dir, "Verification_info.pickle"))    # getting the threshold values to check whether the speaker is verified or not
        if thresholds['EER_thres'] <= sim_score:
          print("Hello %s!"%label.split('_')[0], "Welcome aboard!\n")
        else:
          print("Verification failed! Please try again.\n")

    end_time = time.time()    # End of inference
    np.set_printoptions(precision = 2)
    print("Time taken for inference of %s utterance(s): %0.2fs"%(len(spk_label), end_time-start_time))



"""# Main module
if __name__ == "__main__":
  # Add configuration later
    #DO

  # Training the model
  
  #print("TRAINING SESSION...")
  #Preprocesing the data to obtain mfcc or spectrogram for input to the network
  #ppr.preprocess_data(audio_dir, dev_set_path, train_data_path, 'train')
  #ppr.preprocess_data(audio_dir, eval_set_path, eval_data_path, 'train')
  #train(audio_dir, train_data_path)
      

  # Enrolling the speakers
  #print("ENROLLMENT SESSION...")
  ppr.preprocess_data(audio_dir, enroll_set_path, enroll_data_path, 'enroll')"""