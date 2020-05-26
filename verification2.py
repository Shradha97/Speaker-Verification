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
dev_set_path = '/Users/apple/Desktop/BTP/audio_data_SLR22/list/check_train.txt'
enroll_set_path = '/Users/apple/Desktop/BTP/audio_data_SLR22/list/check_enroll.txt'
eval_set_path = '/Users/apple/Desktop/BTP/audio_data_SLR22/list/check_eval.txt'
test_set_path = '/Users/apple/Desktop/BTP/audio_data_SLR22/list/check_test.txt'    # for inference

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

iteration = 100 # Number of iterations for training
lr = 0.01   # Learning rate
lr_decay_step = 10    # Number of epochs after which the lr will reduce
epochs = 3
N = 4   # Number of speakers in a training batch
M = 5   # Number of utterances per speaker for training
M_enroll = 6  # Number of utterances used for enrollment
M_eval = 7    # Number of utterances taken from speaker for evaluation
one_hot_spk_labels = {} # A dictionary to contain label corresponding to each development speaker file - GLOBAL VARIABLE
#spk_count_dev = 0   # To keep the active count of the development speakers labelled in dictionary while preprocessing - GLOBAL VARIABLE
dev_set_speakers = 180

# Setting model variables
n_filters = 15
kernel_dim = 24
stride = kernel_dim

# For restoring model
model_number = 2




# Function to read .wav audio file
def read_file(audio_file, sampling_rate = 16000):
  audio, sampling = librosa.load(audio_file, sr = sampling_rate)
  audio_trimmed, index = librosa.effects.trim(audio)
  return audio_trimmed, sampling



# Function to do FFT Normalization of the audio
def fft_normalize(audio):
  audio = np.fft.fft(audio)
  audio = audio*100/np.max(abs(audio))  # FFT Normalization
  audio = np.real(np.fft.ifft(audio))
  return audio



# Module to read path to audio files and finally return the readfiles
def read_files(audio_dir, path, session):
  if session == 'train':
    # Reading from the development set
    print("Reading files from the Development Set...")
    print("Path: ", path, "\n")
    with open(path, 'r') as f:
      file_path = f.readlines()
  elif session == 'enroll':
    # Reading from the enrollment set
    print("Reading files from the Enrollment Set...")
    print("Path: ", path, "\n")
    with open(path, 'r') as f:
      file_path = f.readlines()
  elif session == 'evaluate':
    # Reading from the evaluation set
    print("Reading files from the Evaluation Set...")
    print("Path: ", path, "\n")
    with open(path, 'r') as f:
      file_path = f.readlines()
  elif session == 'inference':
    # Reading from the test set
    print("Reading files from the Test Set...")
    print("Path: ", path, "\n")
    with open(path, 'r') as f:
      file_path = f.readlines()

  tmp = []
  audio_fileS = []
  normalized_audioS = []
  sound_fileS = []

  for path in file_path:
    file_path_new = os.path.join(audio_dir, path.strip().rstrip("\n"))   # E.g: /content/drive/My Drive/BTP_Speaker_Verification/audio_data_SLR22/enroll/F101_000.wav
    tmp = path.strip().split('/')                                   # E.g: [enroll, F101_000.wav]
    sound_file = path.strip().strip(tmp[0]+"/")
    #print(sound_file)                        # E.g: F101_000.wav
    sound_fileS.append(sound_file)

    # Reading .wav file
    print("Reading file %s"%tmp[1])
    audio_file, sampling_freq = read_file(file_path_new)
    audio_fileS.append(audio_file)

    # FFT Normalization
    normalized_audio = fft_normalize(audio_file)
    normalized_audioS.append(normalized_audio)

    tmp.clear()

  #print(sound_fileS)
  print("Finished reading files.\n\n")

  return audio_fileS, normalized_audioS, sound_fileS, sampling_freq




# Module to get label of the speaker files
def get_label(audio_file, session):
  audio_parts = []
  audio = audio_file.strip('.wav')
  audio_parts = audio.split('_')

  if session == 'inference':    # Just for testing
    audio_parts[0] = audio_parts[0]+'_'+audio_parts[2]

  return audio_parts[0]
  """  
  if session == 'test':
    return audio_parts[0]
  else:
    audio_parts[1] = int(audio_parts[1])
    return audio_parts[0], audio_parts[1]"""




# Class to remove silences from the audio fragments
class VoiceActivityDetector():
  '''Use signal energy to detect voice activity in wav file'''

  def __init__(self, input_wav_file, file_name, s_freq):
    self._get_audio(input_wav_file, file_name, s_freq)._convert_to_mono()
    self.sample_window = 0.025 # 25 ms
    self.sample_overlap = 0.01 # 10 ms      Hop length
    self.speech_window = 0.5 # half a second
    self.speech_energy_threshold = 0.4 # 40% of energy in voice band
    self.speech_start_band = 300
    self.speech_end_band = 3000

  def _get_audio(self, audio_file, file_name, s_freq):
    self.rate = s_freq
    self.data = audio_file
    self.channels = len(self.data.shape)
    self.filename = file_name
    return self

  def _convert_to_mono(self):
    if self.channels == 2:
      self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
      self.channels = 1
    return self

  def _calculate_frequencies(self, audio_data):
    data_freq = librosa.core.fft_frequencies(sr = 16000, n_fft=len(audio_data))
    #data_freq = np.fftfreq(len(audio_data), 1.0/self.rate)
    data_freq = data_freq[1:]
    return data_freq

  def _calculate_amplitude(self, audio_data):
    data_ampl = np.abs(np.fft.fft(audio_data))
    data_ampl = data_ampl[1:]
    return data_ampl

  def _calculate_energy(self, data):
    data_amplitude = self._calculate_amplitude(data)
    data_energy = data_amplitude ** 2
    return data_energy

  def _connect_energy_with_frequencies(self, data_freq, data_energy):
    energy_freq = {}
    for (i, freq) in enumerate(data_freq):
      if abs(freq) not in energy_freq:
        energy_freq[abs(freq)] = data_energy[i] * 2
    return energy_freq

  def _calculate_normalized_energy(self, data):
    data_freq = self._calculate_frequencies(data)
    data_energy = self._calculate_energy(data)
    energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)
    return energy_freq

  def _sum_energy_in_band(self, energy_frequencies, start_band, end_band):
    sum_energy = 0
    for f in energy_frequencies.keys():
      if start_band < f and f < end_band:
        sum_energy += energy_frequencies[f]
    return sum_energy

  def _median_filter(self, x, k):
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

  def _smooth_speech_detection(self, detected_windows):
    median_window = int(self.speech_window/self.sample_window)
    if median_window % 2 == 0:
      median_window = median_window - 1
      median_energy = self._median_filter(detected_windows[:, 1], median_window)
      return median_energy

  def _no_silence(self, speech_time, audio, speech_without_silence):
    for part in speech_time:
        speech_without_silence = np.append(speech_without_silence, audio[int(part['speech_begin']*self.rate):int(part['speech_end']*self.rate)])
    return speech_without_silence

  def convert_windows_to_readable_labels(self, audio, detected_windows):
    speech_time = []
    is_speech = 0
    speech_without_silence = []
    for window in detected_windows:
      if(window[1] == 1.0 and is_speech == 0):
        is_speech = 1
        speech_label={}
        speech_time_start = window[0] / self.rate
        speech_label['speech_begin'] = speech_time_start
      if(window[1]==0.0 and is_speech==1):
        is_speech = 0
        speech_time_end = window[0] / self.rate
        speech_label['speech_end'] = speech_time_end
        speech_time.append(speech_label)
    speech_without_silence = self._no_silence(speech_time, audio, speech_without_silence)
    return speech_without_silence

  def detect_speech(self):
    '''Detects speech regions based on ratio between speech band energy
    and total energy.
    Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech).'''

    detected_windows = np.array([])
    sample_window = int(self.rate*self.sample_window)
    sample_overlap = int(self.rate*self.sample_overlap)
    data = self.data
    sample_start = 0
    start_band = self.speech_start_band
    end_band = self.speech_end_band
    while(sample_start < (len(data)-sample_window)):
        sample_end = sample_start+sample_window
        if sample_end >= len(data):
            sample_end = len(data)-1
        data_window = data[sample_start:sample_end]
        energy_freq = self._calculate_normalized_energy(data_window)
        sum_voice_energy = self._sum_energy_in_band(energy_freq, start_band, end_band)
        sum_full_energy = sum(energy_freq.values())
        speech_ratio = sum_voice_energy/sum_full_energy

        # Going with the hypothesis that when there is a speech sequence we have ratio of energies more than threshold
        speech_ratio = speech_ratio > self.speech_energy_threshold
        detected_windows = np.append(detected_windows, [sample_start, speech_ratio])
        sample_start += sample_overlap
        detected_windows = detected_windows.reshape(int(len(detected_windows)/2), 2)
        detected_windows[:, 1] = self._smooth_speech_detection(detected_windows)
    return detected_windows




# Module to remove silence from the audio fragment
def remove_silence(audio_file, audio_file_name, s_freq, data_path, session):
  #print("Removing the silent parts from the audio...")

  v = VoiceActivityDetector(audio_file, audio_file_name, s_freq) # Inputting the .wav file to the VAD
  raw_detection = v.detect_speech()
  new_speech = v.convert_windows_to_readable_labels(audio_file, raw_detection)

  #print("Silence removal process completed.\n")
  return new_speech # Returning new audio with removed silence





def break_audio_frames(audio):
    feature_parts = np.matrix([]).reshape((0, nmfcc))
    if np.shape(audio)[1] < nmfcc:       # We need same number of frames and n_mels in our features
        while np.shape(audio)[1] < nmfcc:
            audio = np.concatenate((audio, audio), axis = 1)
    for i in range(0, np.shape(audio)[1], nmfcc):
        if i + nmfcc > np.shape(audio)[1]:              # If duing breaking number of remaining frames are less than nmfcc then attach the first few frames to compensate for the required number of frames
            end = (i + nmfcc) - np.shape(audio)[1]
            tmp = np.hstack((audio[:, i:np.shape(audio)[1]], audio[:, 0: end]))
            feature_parts = np.concatenate([feature_parts, tmp], axis = 0)
            break
        else:
            feature_parts = np.concatenate([feature_parts[:, 0:nmfcc], audio[:, i:i+nmfcc]], axis = 0)
    features_matrix = np.reshape(np.array(feature_parts), (-1, nmfcc, nframes))
    return features_matrix




# Module to extract mfccs from the audio
def get_mfcc(audio, norm_audio, audio_file, speaker_label, s_freq, data_path, session):
  #print("Extracting text independent features from the utterance...")
  audio_data = remove_silence(norm_audio, audio_file, s_freq, data_path, session) # Removing the silent regions of the audio

  # Dividing the utterances into segments of length = sv_mfcc_frame (505ms) -- As later during shuffling the batch we need to consider batches from [415, 515] ms length
  speech = np.abs(librosa.stft(audio_data, window = "hamming", n_fft = nfft, win_length = int(window*s_freq), hop_length = int(hop_train*s_freq)))**2
  speech = librosa.feature.melspectrogram(S = speech, y = audio_data, n_mels = nmels)
  feats = librosa.feature.mfcc(S = librosa.power_to_db(speech), n_mfcc = nmfcc)

  # Dividing into partial utterances
  utterance_mfcc = break_audio_frames(feats)        # To keep the mfcc values of partial utterances. Shape: (number of partial utterances, n_mfcc, n_frames)
  return utterance_mfcc





# Module to get one hot speaker embeddings
def one_hot_speaker_label(index, depth = 3):	# Here 180 is the number of speakers in the train set
    a = tf.Variable(index)
    a_one_hot = tf.one_hot(a, depth)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(a_one_hot)
        return out          # Output: [1 x depth]





# Module to save processed utterance mfccs to a new path
def save_mfcc_to_path(utterances_spec, speaker, audio_dir, data_path, spk_count, session):
  #utterances_spec = np.array(utterances_spec) # Storing list of mel-coeff as a numpy array
  print("Shape of %s data is:"%speaker, utterances_spec.shape)
  print("Saving %s data to:"%speaker, os.path.join(os.path.join(os.path.join(audio_dir, data_path), "Speaker_test"), speaker+".npy"))
  np.save(os.path.join(os.path.join(os.path.join(audio_dir, data_path), "Speaker_test"), speaker+".npy"), utterances_spec)  # Saving the mel-coefficients of the utterances in .npy file
  if session == 'train':
    # Creating key-value pairs for original label of the speaker and for one-hot encoding
    global one_hot_spk_labels
    if speaker not in one_hot_spk_labels:
      one_hot_spk_labels[speaker] = [spk_count, one_hot_speaker_label(spk_count)]






# Module to save the processed utterance mfccs of the speakers providing their voices dynamically
def save_mfcc_to_test_path(audio_mfcc, speaker, audio_dir, data_path):
  print("Shape of %s data is:"%speaker, audio_mfcc.shape)
  print("Saving %s data to:"%speaker, os.path.join(os.path.join(audio_dir, data_path), speaker+".npy"), '\n')
  np.save(os.path.join(os.path.join(audio_dir, data_path), speaker+".npy"), audio_mfcc)  # Saving the mel-coefficients of the utterances in .npy file





# Module to get mfcc of the audios
def preprocess_data(audio_dir, set_path, data_path, session):
    prev_speaker = "null"
    count = 0
    spk_count = 0       # To keep active count of developement speakers in the dictionary

    audio_list, norm_audio_list, audio_file_list, sampling_freq = read_files(audio_dir, set_path, session)
    os.makedirs(os.path.join(audio_dir, data_path), exist_ok = True) # Making folder to save the corresponding session files

    if session != 'inference':
        os.makedirs(os.path.join(os.path.join(audio_dir, data_path), "Speaker_test"), exist_ok = True)  # Making a speaker folder within the directory to have preprocessed speaker files

    print("PREPROCESSING THE SPEAKER FILE(S)...\n")
    for (audio, norm_audio, audio_file) in zip(audio_list, norm_audio_list, audio_file_list):
        if session  != 'inference':
            speaker_label = get_label(audio_file, session)  # Getting the speaker label: F or M + speaker number and the utterance number
            if speaker_label != prev_speaker:
                if prev_speaker != "null": 			# To imply a change in the speaker label, shouldn't be the first speaker
                    print("Finished processing speaker %s."%prev_speaker)
                    save_mfcc_to_path(utterances_spec, prev_speaker, audio_dir, data_path, spk_count, session)  # Here utterances_spec is a 2D array with the inner arrays containing mel-coefficients corresponding to each utterance
                    spk_count = spk_count + 1
                    print("\n")

                utterances_spec = np.empty((0, nmfcc, nframes))            # Resetting the list for the new speaker
                print("Processing speaker %s..."%speaker_label)
            print("Processing file: ", audio_file)
            prev_speaker = speaker_label
            count = count + 1

            # Choose either spectrogram or mfcc
            utterances_spec = np.concatenate((utterances_spec, get_mfcc(audio, norm_audio, audio_file, speaker_label, sampling_freq, data_path, session)), axis = 0)              # To work with mel-coefficients

            # For the last utterance for the last speaker
            if count == len(audio_list):
                print("Finished processing speaker %s."%prev_speaker)
                save_mfcc_to_path(utterances_spec, prev_speaker, audio_dir, data_path, spk_count, session)
                print("\n")
        elif session == 'inference':
            speaker_label = get_label(audio_file, session)  # Getting the speaker label: F or M + speaker number and the utterance number
            print("Processsing speaker %s..."%speaker_label)
            save_mfcc_to_test_path(get_mfcc(audio, norm_audio, audio_file, speaker_label, sampling_freq, data_path, session), speaker_label, audio_dir, data_path)
    if session == 'train':
      #print("Labels: ", one_hot_spk_labels)
      save = open(os.path.join(os.path.join(audio_dir, data_path), "Speaker_one_hot_pickle_test"), 'ab')    # Storing the corresponding speakers' one hot encodings in a pickle file
      pickle.dump(one_hot_spk_labels, save)
      save.close()




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
  model.add(Dense(units = int(60)))
  model.add(BatchNormalization())
  model.add(ReLU())

  # FC 2
  model.add(Dense(units = 60))
  model.add(BatchNormalization())
  model.add(ReLU())
  model.add(Dropout(0.2))

  # FC 3
  model.add(Dense(units = 60))
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
      get_pickle(path, "Speaker_one_hot_pickle_test")
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
def train(audio_dir, data_path, validation_data_path):
  tf.reset_default_graph()   # Reset graph
  total_dev_speakers = len(os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speaker_test")))	# Gives number of target speakers
  

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

  prediction = model.get_layer("dense_3").output
  # Instatiating a loss function
  loss_fn = tf.reduce_mean(categorical_crossentropy(labels, prediction))

  # Using Adam optimizer
  train_step = tf.train.AdamOptimizer(lr_tensor).minimize(loss_fn)

  # Defining metrics for accuracy
  tf_metric, tf_metric_update = tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(prediction, 1), name="my_metric")

  # Isolate the variables stored behind the scenes by the metric operation
  running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")

  # Define initializer to initialize/reset running variables
  running_vars_initializer = tf.variables_initializer(var_list=running_vars)

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
      session.run(running_vars_initializer)         # initialize/reset the running variables for accuracy after each epoch

      for it in range(int(total_dev_speakers*50/(2*M))):
        input_batch, spk_label = get_random_batch('train', 2, M, os.path.join(audio_dir, data_path)) # DEFINING N=2 AND M=5!!!!
        _, loss_cur, summary = sess.run([train_step, loss_fn, merged], feed_dict = {'conv2d_input:0': input_batch, labels: spk_label, lr_tensor: lr*lr_factor})

        # Update the running variables on new batch of samples to obtain the accuracy
        #feed_dict={tf_label: labels[i], tf_prediction: predictions[i]}
        sess.run(tf_metric_update, feed_dict={labels: spk_label})

        loss_acc +=loss_cur   # accumulated loss for each of the 10 iterations

        if iteration%10 == 0:
          writer.add_summary(summary, iteration)  # Writing at tensorboard

        iteration = iteration + 1
        print("Loss is: ", loss_cur/100)

      # Obtaining the accuracy after each epoch
      accuracy = sess.run(tf_metric)

      # Doing the validation
      input_batch, spk_label = get_random_batch('train', 2, M, os.path.join(audio_dir, validation_data_path))
      validation_loss = sess.run()

      # Recording loss at the end of the epoch
      print("Epoch : %d        Loss: %.4f        Train accuracy: %.4f" % (epoch,loss_acc/100, accuracy))
      loss_acc = 0    # Resetting accumulated loss

      if (epoch + 1)%lr_decay_step == 0:
        lr_factor /= 2      # lr decay
        print("Learning rate is decayed! Current lr: ", lr*lr_factor)
      if (epoch + 1)%1 == 0:
        save_path = saver.save(sess, os.path.join(audio_dir, "Check_Point/model"), global_step = epoch//1)  # Saving the model after every 10 epochs
        print("Model saved in path: %s" % save_path)
        
  


# Module to normalize the input        
def normalize(inp):
  return inp/tf.sqrt(tf.reduce_sum(tf.math.square(inp), axis = -1, keepdims = True)+1e-6)




# Module to obtain the averaged embedding over a few considered utterances
def save_av_embedding(embeddings, n, m, speakers, data_path):
  ''' Saving the averaged enrollment utterances into a pickle file.'''

  file_path = os.path.join(data_path, "Speaker_embeddings_test.pickle")
  data = {} # The dictionary to store 
  #print("Shape: ", np.shape(embeddings))


  for count, spk in enumerate(speakers):
    embedding = normalize(tf.reduce_mean(embeddings[count], axis = 0))  # Obtaining the normalized element wise average of different utterance embeddings
    

    with tf.Session() as sess:
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
  spk_list = [f for f in os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speaker_test")) if not f.startswith('.')]   # To remove unnecessary hidden files

  speakers_to_enroll = len(spk_list) # Gives number of unenrolled processed speakers present inside the Speaker folder

  #tf.reset_default_graph()
  
  # Redefining the same model so that the old weights can be loaded into it
  model = create_model()
  
  # Routine to get the speaker embeddings 
  embed = normalize(model.get_layer("re_lu_2").output)   # the embedding shape: [batch_size, no. of output units in the FC3 after activation]
  #saver = tf.train.Saver(var_list = tf.global_variables())
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Loading the model from checkpoint
    print("Restoring the Trained Model...")
    try:
      saver = tf.train.import_meta_graph(os.path.join(audio_dir, 'Check_Point/'+'model-'+str(model_number)+'.meta'))
      saver.restore(sess, tf.train.latest_checkpoint(os.path.join(audio_dir, "Check_Point")))
      print("Model loaded from %s\n"%os.path.join(audio_dir, 'Check_Point/'+'model-'+str(model_number)+'.meta'))
    except ValueError:
      raise ValueError("The required model does not exist! Check model path or model number.\n")

    input_batch, spk_label = get_random_batch(session, speakers_to_enroll, M_enroll, os.path.join(audio_dir, data_path), False)
    #print("Input batch: ", np.shape(input_batch))
    #print("Labels: ", np.shape(spk_label))

    # Copying the enrolled speakers into the enrollment_done folder now...while actual running move into the folder instead of copying into it
    for f in os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speaker_test")):      #FOR COPYING
      if not f.startswith('.'):
        print("Saving %s "%f.strip(".npy")+"input data to %s"%os.path.join(os.path.join(audio_dir, data_path), "Enrollment_done"))
        shutil.move(os.path.join(os.path.join(os.path.join(audio_dir, data_path), "Speaker_test"), f), os.path.join(os.path.join(audio_dir, data_path), "Enrollment_done"))


    print("\nENROLLING THE SPEAKER(S)...")
    spk_embeddings = sess.run(embed, feed_dict = {'conv2d_input:0': input_batch})   # Getting the speaker embedding
    #print("Embeddings: ", np.shape(spk_embeddings))

    #spk_embeddings = spk_embeddings.reshape([speakers_to_enroll, M_enroll, -1])   # Reshaping to be of the shape: [Number to enrolled speakers, number of utterances per speaker, number of embedding features]
    #print("Reshaped: ", np.shape(spk_embeddings))
    save_av_embedding(sess.run(tf.reshape(spk_embeddings, shape = [speakers_to_enroll, M_enroll, -1])), speakers_to_enroll, M_enroll, spk_label, os.path.join(audio_dir, data_path))

    
    print("\nEnrollment completed.\n")
    """
    for f in os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Speaker_test")):      #FOR MOVING
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
  enrolled_tensor = tf.placeholder(shape = [None], dtype = tf.float32)
  test_tensor = tf.placeholder(shape = [None], dtype = tf.float32)

  sim = get_similarity(enrolled_tensor, test_tensor)

  #print("Test speaker shape: ", np.shape(test_spk))
  enrolled_speakers = get_spk_embedding(file_path)  # To get embeddings of all the speakers for testing
  S = np.empty((speaker_num, utter_num, len(enrolled_speakers.keys())))

  #print("Enrollment speaker: ", enrolled_speakers.keys())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(speaker_num):
      for j in range(utter_num):
        for k, enr_spk in enumerate(enrolled_speakers.keys()):
          #print("Enrolled shape: ", np.shape(enrolled_speakers[enr_spk]))
          #print("Shape: ", np.shape(test_spk[i, j, :]))
          #print("Test shape: ", np.shape(test_spk[i, j, :]))
          #print("Similarity: ", sess.run(sim, feed_dict = {enrolled_tensor: enrolled_speakers[enr_spk], test_tensor: test_spk[i, j, :]}))
          S[i, j, k] = sess.run(sim, feed_dict = {enrolled_tensor: enrolled_speakers[enr_spk], test_tensor: test_spk[i, j, :]})
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
def evaluate(audio_dir, data_path):
  #tf.reset.default_graph()
  spk_list = [f for f in os.listdir(os.path.join(os.path.join(audio_dir, data_path), "Enrollment_done")) if not f.startswith('.')]   # To remove unnecessary hidden files
  spks_to_evaluate = len(spk_list) # Gives number of speakers to be evaluated

  #eval_embed = tf.placeholder(shape = [spks_to_evaluate, M_eval, None], dtype = tf.float32) # The batch size will be 1(Number of speakers)*7(Number of utterances per speaker)
  #spk_embed = tf.placeholder(shape = [spks_to_evaluate*M_eval, None], dtype = tf.float32)

  # Redefining the same model so that the old weights can be loaded into it
  model = create_model()

  # Routine to get the test speaker embeddings 
  test_embed = normalize(model.get_layer("re_lu_2").output)   # The embedding shape: [batch_size, no. of output units in the FC3]
  #similarity_matrix = get_similarity_matrix(, spks_to_evaluate, M_eval, os.path.join(os.path.join(audio_dir, data_path), "Speaker_embeddings_test.pickle"))   # Second last 2 arguments represent: Number of enrolled speakers, number of test utterances per speaker
  #saver = tf.train.Saver(var_list = tf.global_variables())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Loading the model from checkpoint
    print("Restoring the Trained Model...")
    try:
      saver = tf.train.import_meta_graph(os.path.join(audio_dir, 'Check_Point/'+'model-'+str(model_number)+'.meta'))
      saver.restore(sess, tf.train.latest_checkpoint(os.path.join(audio_dir, "Check_Point")))
      print("Model loaded from %s\n"%os.path.join(audio_dir, 'Check_Point/'+'model-'+str(model_number)+'.meta'))
    except ValueError:
      raise ValueError("The required model does not exist! Check model path or model number.\n")

    time1 = time.time() # for check inference time
    input_batch, spk_label = get_random_batch('evaluate', spks_to_evaluate, M_eval, os.path.join(os.path.join(audio_dir, data_path)), False)

    #print("Input batch: ", np.shape(input_batch))
    #print("Labels: ", np.shape(spk_label))

    spk_embedding = sess.run(tf.reshape(test_embed, shape = [spks_to_evaluate, M_eval, -1]), feed_dict = {'conv2d_input:0': input_batch})   # Getting the test speaker embedding
    #print("Shape of test speaker embeddings: ", np.shape(spk_embedding))  
    #print("Speakers are: ", spk_label)

    S = get_similarity_matrix(spk_embedding, spks_to_evaluate, M_eval, os.path.join(os.path.join(audio_dir, data_path), "Speaker_embeddings_test.pickle"))    # Getting the similarity score matrix
    print(S)    # Similarity matrix
    time2 = time.time()
    np.set_printoptions(precision=2)
    print("inference time for %d utterences : %0.2fs"%(M_eval*spks_to_evaluate, time2-time1))


    # Calculating EER
    diff = 1; EER=0; EER_thres = 0; EER_FAR=0; EER_FRR=0

    # Calculating FAR and FRR
    for thres in [0.01*i+0.5 for i in range(50)]:
      S_thres = S > thres
      #print("Threshold is: ", thres)
      #print("S_thres is: ", S_thres)

      # False Acceptance Ratio = False Acceptance/ mismatched population (enroll speaker != verification speaker)
      FAR = sum([np.sum(S_thres[i]) - np.sum(S_thres[i,:,i]) for i in range(spks_to_evaluate) for i in range(spks_to_evaluate)])/(spks_to_evaluate-1)/M_eval/spks_to_evaluate
      #print("FAR is: ", FAR)

      # False Reject Ratio = False Reject/ matched population (enroll speaker = verification speaker)
      FRR = sum([M_eval-np.sum(S_thres[i][:,i]) for i in range(spks_to_evaluate)])/M_eval/spks_to_evaluate
      #print("FRR is: ", FRR)

      # Save threshold when FAR = FRR (=EER)
      #print("Diff: ", diff)
      #print("New diff: ", abs(FAR-FRR))

      if diff> abs(FAR-FRR):
        diff = abs(FAR-FRR)
        EER = (FAR+FRR)/2
        EER_thres = thres
        EER_FAR = FAR
        EER_FRR = FRR

    print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)"%(EER,EER_thres,EER_FAR,EER_FRR))
    print("Saving the threshold values to ", os.path.join(audio_dir, "Verification_info.pickle", '\n'))
    save_verification_info(os.path.join(audio_dir, "Verification_info.pickle"), EER, EER_thres, EER_FAR, EER_FRR)






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
  print("speakers: ", spks_to_infer)

  test = tf.placeholder(shape = [spks_to_infer*1, nmfcc, None, 1], dtype = tf.float32)  # The batch size will be 1(Number of speakers)*1(Number of utterances per speaker) for testing
  spk_embed = tf.placeholder(shape = [None], dtype = tf.float32)       
  enrolled_embed =  tf.placeholder(shape = [None], dtype = tf.float32) # Will take in speaker's enrollment embedding as the input 

  # Redefining the same model so that the old weights can be loaded into it
  model = create_model()
  
 # Routine to get the test speaker embeddings 
  test_embed = normalize(model.get_layer("re_lu_2").output)   # The embedding shape: [batch_size, no. of output units in the FC3]
  similarity = get_similarity(enrolled_embed, spk_embed)      # Routine to calculate cosine similarity
  #saver = tf.train.Saver(var_list = tf.global_variables())
  enrolled_embed_path = os.path.join(os.path.join(audio_dir, enroll_path), "Speaker_embeddings_test.pickle")

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Loading the model from checkpoint
    try:
      saver = tf.train.import_meta_graph(os.path.join(audio_dir, 'Check_Point/'+'model-'+str(model_number)+'.meta'))
      saver.restore(sess, tf.train.latest_checkpoint(os.path.join(audio_dir, "Check_Point")))
      print("Model loaded from %s\n"%os.path.join(audio_dir, 'Check_Point/'+'model-'+str(model_number)+'.meta'))
    except ValueError:
      raise ValueError("The required model does not exist! Check model path or model number.\n")

    # Return a single similarity score after verification
    # Also measuring the inference time taken
    start_time = time.time()  # Beginning of inference

    input_data, spk_label = get_random_batch('inference', spks_to_infer, 1, os.path.join(audio_dir, test_path), False)
    #print("Data shape:", np.shape(input_data))
    #print("label: ", np.shape(spk_label))

    spk_embedding = sess.run(tf.reshape(test_embed, shape = [spks_to_infer, -1]), feed_dict = {'conv2d_input:0': input_data})   # Getting the test speaker embedding
    print("Shape of test speaker embedding: ", np.shape(spk_embedding))  

    enrolled_spk_embed_data = get_spk_embedding(enrolled_embed_path)    # Getting the enrollment speaker embeddings
    #print("Enrollment data: ",  enrolled_spk_embed_data)

    enrolled_spk_embed = np.empty((0, np.shape(spk_embedding)[1]))
    for index, label in enumerate(spk_label):
      if label.split('_')[0] not in enrolled_spk_embed_data.keys():
        print("%s is an invalid identity"%label.split('_')[0])
      else:
        #print("Hello: ", np.shape(enrolled_spk_embed_data[label.split('_')[0]]))
        #print("Hi: ", np.shape(spk_embedding[index]))
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
    print("Time taken for inference: %0.2fs"%(end_time-start_time))






# Main module
if __name__ == "__main__":
  # Add configuration later
    #DO
  # Training the model
  print("TRAINING SESSION...")
  # Preprocesing the data to obtain mfcc or spectrogram for input to the network
  #preprocess_data(audio_dir, dev_set_path, train_data_path, 'train')
  preprocess_data(audio_dir, eval_set_path, eval_data_path, 'evaluate')
  train(audio_dir, train_data_path, eval_data_path)

  # Enrolling the speakers
  #print("ENROLLMENT SESSION...")
  #preprocess_data(audio_dir, enroll_set_path, enroll_data_path, 'enroll')
  #enroll(audio_dir, enroll_data_path, 'enroll')

  # Evaluating the model
  #print("EVALUATION SESSION...")
  #preprocess_data(audio_dir, eval_set_path, eval_data_path, 'evaluate')
  #enroll(audio_dir, eval_data_path, 'enroll')
  #evaluate(audio_dir, eval_data_path)


  # Infering from the model
  #print("INFERENCE SESSION...")
  #preprocess_data(audio_dir, test_set_path, test_data_path, 'inference')
  #inference(audio_dir, enroll_data_path, test_data_path)

