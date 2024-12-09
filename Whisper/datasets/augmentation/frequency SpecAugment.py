import numpy as np
import librosa

# Masking frequency bandwidth
MASKING_FREQUENCY_BANDWIDTH = 20  # Example masking bandwidth

def apply_spec_augment_frequency_masking(y, sr, freq_bandwidth):
    """
    Applies SpecAugment-style frequency masking to the audio signal's spectrogram.
    """
    # Convert to spectrogram
    stft = librosa.stft(y)
    spectrogram = np.abs(stft)

    # Randomly select a start position for frequency masking
    num_freq_bins = spectrogram.shape[0]
    mask_start = np.random.randint(0, num_freq_bins - freq_bandwidth)
    spectrogram[mask_start:mask_start + freq_bandwidth, :] = 0

    # Reconstruct audio signal from the masked spectrogram
    masked_stft = spectrogram * np.exp(1j * np.angle(stft))
    y_masked = librosa.istft(masked_stft)
    return y_masked

def apply_augmentation(y, sr):
    """
    Applies frequency masking and returns the masked audio data with a suffix.
    """
    y_masked = apply_spec_augment_frequency_masking(y, sr, MASKING_FREQUENCY_BANDWIDTH)
    return [(y_masked, "spec_augment_frequency_masked")]