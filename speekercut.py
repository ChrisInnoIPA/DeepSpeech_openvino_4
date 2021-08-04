from pydub import AudioSegment
from pydub.silence import split_on_silence

index = 1

def get_wav_make(dataDir,targetDir, start_audiotime, end_audiotime):
    global index
    sound= AudioSegment.from_wav(dataDir)
    cut_wav = sound[start_audiotime * 1000:end_audiotime * 1000]   #以毫秒为单位截取[begin, end]区间的音频
    cut_wav.export(targetDir + str(index) + ".wav", format='wav')   #存储新的wav文件
    index += 1
    print(targetDir + str(index) + ".wav")