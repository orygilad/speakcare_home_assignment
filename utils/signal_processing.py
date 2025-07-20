import torch
import scipy.signal as signal



def low_pass_filter(waveform: torch.Tensor, sample_rate: int, cutoff_freq: float) -> torch.Tensor:
    """
    Applies a low-pass Butterworth filter to the input waveform.

    Args:
        waveform: Tensor of shape [1, num_samples] (mono)
        sample_rate: Sampling rate in Hz
        cutoff_freq: Cutoff frequency in Hz

    Returns:
        Filtered waveform of the same shape
    """
    assert waveform.dim() == 2 and waveform.shape[0] == 1, "Expected mono waveform [1, num_samples]"

    nyquist = sample_rate / 2.0
    norm_cutoff = cutoff_freq / nyquist

    b, a = signal.butter(N=32, Wn=norm_cutoff, btype='low', analog=False)
    filtered = signal.lfilter(b, a, waveform.squeeze(0).numpy())
    return torch.tensor(filtered, dtype=waveform.dtype).unsqueeze(0)


def compute_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    duration_seconds: float,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400
) -> torch.Tensor:
    """
    Computes the STFT magnitude spectrogram (in dB) for a segment of the waveform.

    Args:
        waveform: Tensor of shape [1, num_samples]
        sample_rate: Sampling rate in Hz
        duration_seconds: Duration of the signal (from start) to include
        n_fft: FFT window size
        hop_length: Hop length
        win_length: Window length

    Returns:
        Log-magnitude spectrogram tensor of shape [freq_bins, time_frames]
    """
    assert waveform.dim() == 2 and waveform.shape[0] == 1, "Expected mono waveform [1, num_samples]"

    max_samples = int(duration_seconds * sample_rate)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        return_complex=True
    )

    magnitude = stft.abs().squeeze(0)  # [freq_bins, time_frames]
    db_spec = 20 * torch.log10(magnitude + 1e-9)
    return db_spec
