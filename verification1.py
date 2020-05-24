from __future__ import division
import os
import math
import time
import re

#from scipy.fftpack import dct
import tensorflow as tf
#print(tf.__version__)
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import librosa


# For the training model
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.python.keras.layers.advanced_activations import PReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

# For enrollment and testing
import json

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
enroll_data_path = 'enroll_data'
eval_data_path = 'eval_data'
test_data_path = 'test_data'

# Variables for training and testing set division
#total_dev_speakers = len(os.listdir(dev_set_path)) # Send this for training
#total_enroll_speakers = len(os.listdir(enroll_set_path)) # Send this for enrollment
#total_test_speakers = len(os.listdir(test_set_path)) # Send this for testing
#dev_set_speakers = (total_speaker//10) * 8  # Splitting total development data into 90% train and 10% test data

# Setting variables
'''Taking smaller number of frames while taking spectrogam because
  even lesser number of spectrogram frames will give more information.'''
sv_spec_frame = 180   # Max. frame number of utterances of TI_SV(in ms) for spectrogram
sv_mfcc_utter = 505 # Max. utterance length of TI_SV(in ms) for mfcc
hop_train = 0.01   # Hop size(ms)
window = .025 # Window length(ms)
nfft = 512 # FFT kernel size
nmels = 48 # No. of mels for mel-spectrogram
nmfcc = 48 # No. of mel coefficients
nframes = nmfcc # No. of frames per partial utterance

iteration = 100 # Number of iterations for training
lr = 0.01   # Learning rate
lr_decay_step = 10    # Number of epochs after which the lr will reduce
N = 4   # Number of speakers in a training batch
M = 5   # Number of utterances per speaker for training
M_enroll = 6  # Number of utterances used for enrollment
spk_labels = {} # A dictionary to contain label corresponding to each speaker file - GLOBAL VARIABLE
#spk_count_dev = 0   # To keep the active count of the development speakers labelled in dictionary while preprocessing - GLOBAL VARIABLE
threshold = 1   # Highest threshold ratio - GLOBAL VARIABLE

# Setting model variables
n_filters = 411
kernel_dim = 24
stride = kernel_dim




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
    f.close()
  elif session == 'enroll':
    # Reading from the enrollment set
    print("Reading files from the Enrollment Set...")
    print("Path: ", path, "\n")
    with open(path, 'r') as f:
      file_path = f.readlines()
    f.close()
  elif session == 'evaluation':
    # Reading from the evaluation set
    print("Reading files from the Evaluation Set...")
    print("Path: ", path, "\n")
    with open(path, 'r') as f:
      file_path = f.readlines()
    f.close()
  elif session == 'inference':
    # Reading from the test set
    print("Reading files from the Test Set...")
    print("Path: ", path, "\n")
    with open(path, 'r') as f:
      file_path = f.readlines()
    f.close()

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
  if session == 'test':
    return audio_parts[0]
  else:
    audio_parts[1] = int(audio_parts[1])
    return audio_parts[0], audio_parts[1]




# Class to remove silences from the audio fragments
class VoiceActivityDetector():
  '''Use signal energy to detect voice activity in wav file'''

  def __init__(self, input_wav_file, file_name, s_freq):
    self._get_audio(input_wav_file, file_name, s_freq)._convert_to_mono()
    self.sample_window = 0.025 # 25 ms
    self.sample_overlap = 0.01 # 10 ms      Hop length
    self.speech_window = 0.5 # half a second
    self.speech_energy_threshold = 0.6 # 60% of energy in voice band
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





#Module to break the audio into smaller parts
def break_audio(audio, s_freq):
  utter_len = sv_mfcc_utter * s_freq  # Min allowed length of the utterance

  sample_start = 0
  audio_parts = []  # Containing broken audio parts

  # Doing necessary padding if the length of the passed utterance is smaller than the allowed length
  if len(audio) < utter_len:
    while len(audio) < utter_len:
      audio = np.concatenate((audio, audio), axis = 0)

  while(sample_start <= (len(audio)-utter_len)):
    sample_end = sample_start + utter_len
    if sample_end >= len(audio):  # If the last segment is lesser than the size of the required utterance length, ignore it
      break
    print("Audio part shape is: ", audio[sample_start:sample_end].shape)
    audio_parts.concatenate(audio[sample_start:sample_end], axis = 1)

  return audio_parts



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
  #audio_intervals = break_audio(audio_data)

  speech = np.abs(librosa.stft(audio_data, window = "hamming", n_fft = nfft, win_length = int(window*s_freq), hop_length = int(hop_train*s_freq)))**2
  speech = librosa.feature.melspectrogram(S = speech, y = audio_data, n_mels = nmels)
  feats = librosa.feature.mfcc(S = librosa.power_to_db(speech), n_mfcc = nmfcc)

  # Dividing into partial utterances
  utterance_mfcc = break_audio_frames(feats)        # To keep the mfcc values of partial utterances. Shape: (number of partial utterances, n_mfcc, n_frames)

  return utterance_mfcc





# Module to get one hot speaker embeddings
def one_hot_speaker_label(index, depth = 180):	# Here 180 is the number of speakers in the train set
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

  print("Saving %s data to:"%speaker, os.path.join(os.path.join(os.path.join(audio_dir, data_path), "Speakers"), speaker+".npy"))
  np.save(os.path.join(os.path.join(os.path.join(audio_dir, data_path), "Speakers"), speaker+".npy"), utterances_spec)  # Saving the mel-coefficients of the utterances in .npy file


  if session == 'train':
    # Creating key-value pairs for original label of the speaker and for one-hot encoding
    global spk_labels
    if speaker not in spk_labels:
      spk_labels[speaker] = [spk_count, one_hot_speaker_label(spk_count)]





# Module to get mfcc of the audios
def preprocess_data(audio_dir, set_path, data_path, session):
    prev_speaker = "null"
    count = 0
    spk_count = 0       # To keep active count of developement speakers in the dictionary
    #utterances_spec = np.empty((0, nmfcc, nframes))

    audio_list, norm_audio_list, audio_file_list, sampling_freq = read_files(audio_dir, set_path, session)
    os.makedirs(os.path.join(audio_dir, data_path), exist_ok = True) # Making folder to save the corresponding session files

    if session != 'inference':
        os.makedirs(os.path.join(os.path.join(audio_dir, data_path), "Speakers"), exist_ok = True)  # Making a speaker folder within the directory to have preprocessed speaker files

    print("PREPROCESSING THE SPEAKER FILE(S)...\n")
    for (audio, norm_audio, audio_file) in zip(audio_list, norm_audio_list, audio_file_list):
        if session  != 'inference':
            speaker_label, utterance_count = get_label(audio_file, session)  # Getting the speaker label: F or M + speaker number and the utterance number
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
            save_mfcc_to_test_path(get_mfcc(audio, norm_audio, audio_file, speaker_label, sampling_freq), audio_dir, data_path)




# Main module
if __name__ == "__main__":
  # Add configuration later
    #DO

  """# Training the model
        print("TRAINING SESSION...")
        # Preprocesing the data to obtain mfcc or spectrogram for input to the network
        preprocess_data(audio_dir, dev_set_path, train_data_path, 'train')
      """

  # Enrolling the speakers
  print("Enrollment session")
  preprocess_data(audio_dir, enroll_set_path, enroll_data_path, 'enroll')
  
