import os
import pyaudio
import wave

spk_name = ''

# the file name output you want to record into
def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return 1
    elif reply[0] == 'n':
        return 0
    else:
        return yes_or_no("Please Enter (y/n) ")

while True:
    # DRAW PLOT HERE;
    spk_name = input("Please enter your name: ")
    if(yes_or_no('Is the name %s correct?'%spk_name)):
        break

print("Hi %s", spk_name)
filename = spk_name+'.wav'
print('The filename is: ', filename)

os.makedirs(os.path.join(audio_dir, "RT_enroll"), exist_ok = True)


# number of recordings required
recordings = 6
# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 0
# 16000 samples per second
sample_rate = 16000
record_seconds = 5

print("Please provide "+str(recordings)+" audios for enrollment.")

"""for i in range(recordings):
    # initialize PyAudio object
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Starting recording "+ str(i+1))
    for i in range(int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        # if you want to hear your voice while recording
        # stream.write(data)
        frames.append(data)
    print("Finished recording.")
    # stop and close stream
    stream.stop_stream()
    stream.close()
    # terminate pyaudio object
    p.terminate()
    # save audio file
    # open the file in 'write bytes' mode
    wf = wave.open(os.path.join(os.path.join(audio_dir, "RT_enroll"), filename), "wb")
    # set the channels
    wf.setnchannels(channels)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(sample_rate)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()

    # save the file path
    infile = open(os.path.join(os.path.join(audio_dir, list), "RT_enroll.txt"))
"""