import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
from pydub import AudioSegment
import tempfile

DEFAULT_SR = 22050

def record_audio(duration, sr=DEFAULT_SR):
    print(f"Recording {duration}s at {sr}Hz")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return recording.squeeze()

def save_wav(path, samples, sr=DEFAULT_SR):
    sf.write(path, samples, sr)

def load_audio(path, sr=DEFAULT_SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def extract_features(path, sr=DEFAULT_SR, n_mfcc=13):
    y = load_audio(path, sr)
    if y.size == 0:
        return np.zeros(n_mfcc + 4)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    features = np.concatenate([mfcc_mean, [rms, zcr, rolloff, centroid]])
    return features

def compute_rms_over_time(path, sr=DEFAULT_SR, frame_length=1024, hop_length=512):
    y = load_audio(path, sr)
    if y.size == 0:
        return np.array([0.0])
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return rms

def make_slowmo(path, speed_factor=0.5, out_path=None):
    try:
        audio = AudioSegment.from_file(path)
        new_frame_rate = int(audio.frame_rate * speed_factor)
        slowed = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
        slowed = slowed.set_frame_rate(audio.frame_rate)
        if out_path is None:
            out_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        slowed.export(out_path, format='wav')
        return out_path
    except Exception as e:
        print('make_slowmo failed:', e)
        return path
