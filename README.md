# Speaker-Verification

The "test" folder has the integrated pipeline along with the small data subset. Run the scripts present in that folder.

Train the model first, then do the enrollment. After that you can do evaluation or the inference.

1. To train the model, write the command "python3 main.py --session train". First train the model to get the required model saved in the folder "Saved_model". That would be loaded for the later purposes.

2. To enroll the speakers, write the command "python3 main.py --session enroll".

3. To evaluate the speakers, write the command "python3 main.py --session evaluate".

4. To do inference, write the command "python3 main.py --session inference".

5. Change the set and data paths accordingly in the "config.py" file.

6. You can run the "tfjs_model.py" script like - "python3 tfjs_model.py" to get the model in tensorflowjs compatible form. The converted model will be saved in the "TF-js" folder. Inside "TF-js" folder, the folder "JSON_Models" has the converted model as a "json" file. The file, "model.html" is the script to host the model on the server. I host it on the "google chrome" server. The files, "audio_record.html" and "audio_record.js" are the scripts to record audio from the browser.

7. The "preprocess.py" along with "VAD.py" does the preprocessing of the audio files. The function, "read_files" in "preprocess.py" reads the ".wav" files for the corresponding session and preprocess them, which after getting preprocessed are stored in the form of ".npy" files. These ".npy" files on being loaded are taken up by the model as the input.

8. The "train", "enroll", "evaluate" and "inference" operations are present in the file "verification.py"

9. The output produced by the "enroll" function in the file "verification.py" is stored as a pickle file inside the folder "enrolled_data" which would be created after the script is run.

10. The output produced by the "infer" function in the file "verification.py" is a similarity score with the verdict of verified or not. This function takes in the "embeddings" stored in the pickle file stored in the "enrolled_data" folder and the ".npy" files present in the "test_data" folder created after running the script, which contains the preprocessed audio of the speaker that needs to be tested for inference.


## Requirements

1. tf-1.15
2. Librosa
3. Tensorflowjs
4. pyaudio

