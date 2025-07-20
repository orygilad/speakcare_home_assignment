from utils import audio_utils
from utils import signal_processing
import config
import matplotlib.pyplot as plt
import numpy as np

wav_path = config.SPEAKER_AUDIO_DIR / "Nurse1.wav"
waveform, sr = audio_utils.load_audio(wav_path)

# Apply LPF
filtered = signal_processing.low_pass_filter(waveform, sr, cutoff_freq=config.VOICE_MAX_FREQUENCY)

# Compute and plot spectrogram
spec = signal_processing.compute_spectrogram(filtered, sr, duration_seconds=3.0)

# Get time and frequency axis extents
num_freq_bins, num_frames = spec.shape
freqs = np.linspace(0, sr / 2, num_freq_bins)

plt.figure(figsize=(10, 4))
plt.imshow(
    spec.numpy(),
    origin='lower',
    aspect='auto',
    cmap='magma',
    extent=[0, num_frames, 0, sr // 2],  # time frames (X), freq Hz (Y)
)
plt.title("Filtered Spectrogram (LPF 3kHz)")
plt.xlabel("Time (frames)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="dB")
plt.tight_layout()
plt.show()