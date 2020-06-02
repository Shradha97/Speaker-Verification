import os
import pyaudio
import wave
from config import get_config

cfg = get_config()

spk_name = ''
#audio_dir = '/Users/apple/Desktop/test/audio_data_SLR22'

# the file name output you want to record into
def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return 1
    elif reply[0] == 'n':
        return 0
    else:
        return yes_or_no("Please Enter (y/n) ")


def get_audio():
    while True:
        # DRAW PLOT HERE;
        spk_name = input("Please enter your name: ")
        if(yes_or_no('Are you sure you wanna get enrolled by the name %s'%spk_name)):
            break

    os.makedirs(os.path.join(cfg.audio_dir, "RT_enroll"), exist_ok = True)


    # number of recordings required
    recordings = 6
    # set the chunk size of 1024 samples
    chunk = 1024
    # sample format
    FORMAT = pyaudio.paInt16
    # mono, change to 2 if you want stereo
    channels = 1
    # 16000 samples per second
    sample_rate = 16000
    record_seconds = 10

    print("\nHi %s, please provide "%spk_name+str(recordings)+" audios for enrollment.\n")

    for i in range(recordings):
        # Naming the file
        filename = spk_name+'_'+str(i)+'.wav'
        print('The enrollment filename is: ', filename)

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
        print("STARTING RECORDING "+ str(i+1)+"...")
        for i in range(int(sample_rate / chunk * record_seconds)):
            data = stream.read(chunk)
            # if you want to hear your voice while recording
            # stream.write(data)
            frames.append(data)
        print("FINISHED RECORDING.")
        # stop and close stream
        stream.stop_stream()
        stream.close()
        # terminate pyaudio object
        p.terminate()

        print("Writing the audio to ", os.path.join(os.path.join(cfg.audio_dir, "RT_enroll"), filename), '\n')
        # save audio file
        # open the file in 'write bytes' mode
        wf = wave.open(os.path.join(os.path.join(cfg.audio_dir, "RT_enroll"), filename), "wb")
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

        # save the file path in the list folder
        with open(os.path.join(os.path.join(cfg.audio_dir, 'list'), "RT_enroll.txt"), 'a+') as infile:
    	    infile.seek(0)
    	    data = infile.read(1)
    	    if len(data) > 0:
    	        infile.write('\n')
    	    infile.write('RT_enroll/{}'.format(filename))

        #print('List has RT_enroll/{}\n'.format(filename))

