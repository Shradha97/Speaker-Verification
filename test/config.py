import os
import sys
import argparse

import numpy as np

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

parser = argparse.ArgumentParser()

# get arguments
def get_config():
    config, unparsed = parser.parse_known_args()
    return config


# return bool type of argument
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# return session type
def sess(v):
	if v.lower() in ('Train', 'train', 'Training', 'training'):
		return 'train'
	elif v.lower() in ('Enroll', 'enroll', 'Enrolling', 'enrolling', 'Enrollment', 'enrollment'):
		return 'enroll'
	elif v.lower() in ('Evaluate', 'evaluate', 'Evaluation', 'evaluation'):
		return 'evaluate'
	elif v.lower() in ('Inference', 'inference', 'Infer', 'infer'):
		return 'inference'
	else:
		raise argparse.ArgumentTypeError('Session type expected (Train/Enroll/Evaluate/Inference).')


# Data Preprocess Arguments
data_arg = parser.add_argument_group('Data')
#data_arg.add_argument('--noise_path', type=str, default='./noise', help="noise dataset directory")
data_arg.add_argument('--audio_dir', type=str, default='/Users/apple/Desktop/test/audio_data_SLR22', help="Absolute dataset path")
data_arg.add_argument('--list_path', type=str, default='/Users/apple/Desktop/test/audio_data_SLR22/list', help="Folder having information about paths to different data")
data_arg.add_argument('--dev_set_path', type=str, default='/Users/apple/Desktop/test/audio_data_SLR22/list/check_train.txt', help="Text file containing train set relative paths")
data_arg.add_argument('--val_set_path', type=str, default='/Users/apple/Desktop/test/audio_data_SLR22/list/check_val.txt', help="Text file containing validation set relative paths")
data_arg.add_argument('--enroll_set_path', type=str, default='/Users/apple/Desktop/test/audio_data_SLR22/list/check_enroll.txt', help="Text file containing enrollment set relative paths")
data_arg.add_argument('--RT_enroll_set_path', type=str, default='/Users/apple/Desktop/test/audio_data_SLR22/list/RT_enroll.txt', help="Text file containing enrollment set relative paths taken through microphone")
data_arg.add_argument('--eval_set_path', type=str, default='/Users/apple/Desktop/test/audio_data_SLR22/list/check_eval.txt', help="Text file containing evaluation set relative paths")
data_arg.add_argument('--test_set_path', type=str, default='/Users/apple/Desktop/test/audio_data_SLR22/list/check_test.txt', help="Text file containing inference set relative paths")
data_arg.add_argument('--RT_infer_set_path', type=str, default='/Users/apple/Desktop/test/audio_data_SLR22/list/RT_infer.txt', help="Text file containing inference set relative paths taken through microphone")

data_arg.add_argument('--train_data_path', type=str, default='train_data', help="Train dataset directory")
data_arg.add_argument('--val_data_path', type=str, default='val_data', help="Validation dataset directory")
data_arg.add_argument('--enroll_data_path', type=str, default='enrolled_data', help="Enrollment dataset directory")
data_arg.add_argument('--RT_enroll_data_path', type=str, default='RT_enrolled_data', help="Realtime enrollment dataset directory")
data_arg.add_argument('--eval_data_path', type=str, default='eval_data', help="Evaluation dataset directory")
data_arg.add_argument('--test_data_path', type=str, default='test_data', help="Inference dataset directory")
data_arg.add_argument('--RT_infer_data_path', type=str, default='RT_infer_data', help="Realtime inference dataset directory")



# For preprocessing
preprocess_arg = parser.add_argument_group('Data Preprocessing')
preprocess_arg.add_argument('--nfft', type=int, default=512, help="fft kernel size")
preprocess_arg.add_argument('--window', type=int, default=0.025, help="window length (ms)")
preprocess_arg.add_argument('--hop_train', type=int, default=0.01, help="hop size (ms)")
preprocess_arg.add_argument('--frame_range_low', type=int, default=39, help="min utterance length in terms of frames")
preprocess_arg.add_argument('--frame_range_high', type=int, default=48, help="max utterance length in terms of frames")
preprocess_arg.add_argument('--nmels', type=int, default=48, help="no. of mels for mel-spectrogram")
preprocess_arg.add_argument('--nmfcc', type=int, default=48, help="no. of mel coefficients")
preprocess_arg.add_argument('--nframes', type=int, default=48, help="no. of frames per partial utterance")


# Model Parameters
model_arg = parser.add_argument_group('Model')
model_arg.add_argument('--n_filters', type=int, default=411, help="number of filters used for convolution")
model_arg.add_argument('--kernel_dim', type=int, default=24, help="size of the kernel used for convolution")
model_arg.add_argument('--stride', type=int, default=24, help="strides used for convolution")
model_arg.add_argument('--restore', type=str2bool, default=False, help="restore model or not")
model_arg.add_argument('--model_path', type=str, default='Saved_Model', help="relative model directory to save or load")
model_arg.add_argument('--model_number', type=int, default=2, help="the model number that needs to be restored")
model_arg.add_argument('--num_classes', type=int, default=6, help="the number of classes into which the model has to classify")


# Training Parameters
train_arg = parser.add_argument_group('Training')
train_arg.add_argument('--session', type=sess, required = True, help="session type (train/enroll/evaluate/inference)")
train_arg.add_argument('--N', type=int, default=4, help="number of speakers of training batch")
train_arg.add_argument('--M', type=int, default=5, help="number of utterances per training speaker")
train_arg.add_argument('--lr', type=float, default=1e-4, help="learning rate")
train_arg.add_argument('--lr_decay_step', type=float, default=10, help="number of epochs after which the lr will reduce")
train_arg.add_argument('--epochs', type=float, default=3, help="total number of epochs for training")
train_arg.add_argument('--iteration', type=int, default=100000, help="max iteration")

# Enrollment Parameters
enroll_arg = parser.add_argument_group("Enrollment")
enroll_arg.add_argument('--M_enroll', type=int, default=6, help="number of utterances per enrolling speaker for obtaining the embedding")

# Evaluation Parameters
eval_arg = parser.add_argument_group("Evaluation")
eval_arg.add_argument('--M_eval', type=int, default=7, help="number of utterances per speaker being evaluated")


config = get_config()
print(config)           # print all the arguments
