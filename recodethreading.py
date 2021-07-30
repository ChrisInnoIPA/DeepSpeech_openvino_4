import threading
import datetime
import sounddevice as sd
from scipy.io.wavfile import write


indexx = 1
t_num = 0
mutex = threading.Lock()

def recod_5_time():
        print('start=====>', datetime.datetime.now())
    # while(True):
        mutex.acquire()
        filename = '0001'
        fs=16000
        duration = 3  # seconds
        # print("test===>")
        myrecording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype=np.int16)
        # real_5_time(myrecording)
        print("Recording Audio")
        sd.wait()
        global indexx
        # print("Audio recording complete , Play Audio")
        write(filename + str(indexx) + '.wav', fs, myrecording) 
        indexx += 1
        mutex.release()
        
        # print(myrecording, "myrecording=================>")

        # sd.play(myrecording, fs)
        # sd.wait()
        # print("Play Audio Complete")
        # print('2=====>')
        real_5_time(myrecording, fs)
        print('end=====>', datetime.datetime.now())