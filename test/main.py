from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import os
from config import get_config
import preprocess as ppr
import verification as vs
import audio_record_enroll as enr
import audio_record_infer as inf


cfg = get_config()
tf.reset_default_graph()


# Main module
if __name__ == "__main__":
  # Training the model
  if cfg.session == 'train':
    print("TRAINING SESSION...")
    #Preprocesing the data to obtain mfcc or spectrogram for input to the network
    ppr.preprocess_data(cfg.audio_dir, cfg.dev_set_path, cfg.train_data_path, cfg.session)    # Have to do the seapration into train and validation sets manually
    ppr.preprocess_data(cfg.audio_dir, cfg.val_set_path, cfg.val_data_path, cfg.session)
    vs.train(cfg.audio_dir, cfg.train_data_path, cfg.val_data_path)

  # Evaluating the model
  elif cfg.session == 'evaluate':
    print("EVALUATION SESSION...")
    ppr.preprocess_data(cfg.audio_dir, cfg.enroll_set_path, cfg.enroll_data_path, 'enroll')
    ppr.preprocess_data(cfg.audio_dir, cfg.eval_set_path, cfg.eval_data_path, cfg.session)
    vs.evaluate(cfg.audio_dir, cfg.eval_data_path, cfg.enroll_data_path)  

  # Enrolling the speakers
  elif cfg.session == 'enroll':
    print("ENROLLMENT SESSION...")
    reply = enr.yes_or_no('Do you want to give input through microphone?(y/n): ')
    if(reply):
      enr.get_audio()
      ppr.preprocess_data(cfg.audio_dir, cfg.RT_enroll_set_path, cfg.RT_enroll_data_path, cfg.session)
      vs.enroll(cfg.audio_dir, cfg.RT_enroll_data_path, cfg.session)
    else:
      ppr.preprocess_data(cfg.audio_dir, cfg.enroll_set_path, cfg.enroll_data_path, cfg.session)
      vs.enroll(cfg.audio_dir, cfg.enroll_data_path, cfg.session)

  # Infering from the model
  else:
    print("INFERENCE SESSION...")
    reply = enr.yes_or_no('Do you want to give input through microphone?(y/n): ')
    if(reply):
      inf.get_audio()
      ppr.preprocess_data(cfg.audio_dir, cfg.RT_infer_set_path, cfg.RT_infer_data_path, cfg.session)
      vs.inference(cfg.audio_dir, cfg.RT_enroll_data_path, cfg.RT_infer_data_path)
    else:
      ppr.preprocess_data(cfg.audio_dir, cfg.test_set_path, cfg.test_data_path, cfg.session)
      vs.inference(cfg.audio_dir, cfg.enroll_data_path, cfg.test_data_path)



  


