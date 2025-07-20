import os
os.environ["PATH"] += os.pathsep + r"C:\ProgramData\chocolatey\bin"
from pydub import AudioSegment
import torchaudio
import config
import torchaudio.transforms as T

def convert_m4a_to_wav(input_path, output_path=None):
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".wav"
    audio = AudioSegment.from_file(input_path, format="m4a")
    #audio.export(output_path, format="wav", codec="pcm_s16le")
    audio.export(output_path, format="wav", codec="pcm_s16le", parameters=["-ar", str(config.SAMPLE_RATE)])

    return output_path

def chunk_audio(input_wav_path, chunk_duration_ms=3000):
    audio = AudioSegment.from_wav(input_wav_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i+chunk_duration_ms]
        chunks.append(chunk)
    return chunks
def load_audio(audio_path):
    signal, fs = torchaudio.load(audio_path)

    if fs != config.SAMPLE_RATE:
        #print(f"Skipping expected {config.SAMPLE_RATE}Hz, got {fs}Hz.")
        print(f"Resampling from {fs} Hz → {config.SAMPLE_RATE} Hz")
        resampler = T.Resample(orig_freq=fs, new_freq=config.SAMPLE_RATE)
        signal = resampler(signal)
    return signal,fs
