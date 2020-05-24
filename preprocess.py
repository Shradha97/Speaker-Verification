from __future__ import division
import os
import math
import re

#from scipy.fftpack import dct
import tensorflow as tf
#print(tf.__version__)
import numpy as np
import librosa
import pickle

from VAD import VoiceActivityDetector, remove_silence


#sv_mfcc_utter = 505 # Max. utterance length of TI_SV(in ms) for mfcc
frame_range_low = 39  # Min utterance length in terms of frames
frame_range_high = 48 # Max utterance length in terms of frames
hop_train = 0.01   # Hop size(ms)
window = .025 # Window length(ms)
nfft = 512 # FFT kernel size
nmels = 48 # No. of mels for mel-spectrogram
nmfcc = 48 # No. of mel coefficients
nframes = nmfcc # No. of frames per partial utterance
one_hot_spk_labels = {} # A dictionary to contain label corresponding to each development speaker file - GLOBAL VARIABLE



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
  elif session == 'evaluation':
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
  if session == 'test':
    return audio_parts[0]
  else:
    audio_parts[1] = int(audio_parts[1])
    return audio_parts[0], audio_parts[1]





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
def one_hot_speaker_label(index, depth = 180):	# Here 200 is the number of speakers in the train set
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
    global one_hot_spk_labels
    if speaker not in one_hot_spk_labels:
      one_hot_spk_labels[speaker] = [spk_count, one_hot_speaker_label(spk_count)]





# Module to get mfcc of the audios
def preprocess_data(audio_dir, set_path, data_path, session):
    prev_speaker = "null"
    count = 0
    spk_count = 0       # To keep active count of developement speakers in the dictionary
    #utterances_spec = np.empty((0, nmfcc, nframes))

    audio_list, norm_audio_list, audio_file_list, sampling_freq = read_files(audio_dir, set_path, session)
    os.makedirs(os.path.join(audio_dir, data_path), exist_ok = True) # Making folder to save the corresponding session files

    if session != 'inference':
        os.makedirs(os.path.join(os.path.join(audio_dir, data_path), "Speaker_test"), exist_ok = True)  # Making a speaker folder within the directory to have preprocessed speaker files

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

    if session == 'train':
    	save = open(os.path.join(os.path.join(audio_dir, data_path), "One_hot.pickle"), 'ab')
    	pickle.dump(one_hot_spk_labels, save)
    	save.close()

