from os import path
from pydub import AudioSegment

ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'

AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffmpeg_path

def convert_to_wav(src,dst):
    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")
