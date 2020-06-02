# Speaker-Verification

The "test" folder has the integrated pipeline along with the small data subset. Run the scripts present in that folder.

1. To train the model, write the command "python3 main.py --session train". First train the model to get the required model saved in the folder "Saved_model". That would be loaded for the later purposes.

2. To enroll the speakers, write the command "python3 main.py --session enroll".

3. To evaluate the speakers, write the command "python3 main.py --session evaluate".

4. To do inference, write the command "python3 main.py --session inference".

5. Change the set and data paths accordingly in the "config.py" file.

6. You can run the "tfjs_model.py" script like - "python3 tfjs_model.py" to get the model in tensorflowjs compatible form. The converted model will be saved in the "TF-js" folder.

