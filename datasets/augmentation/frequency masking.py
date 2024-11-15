# import numpy as np
# import librosa

# # Frequency range for masking
# MASKING_FREQUENCY_RANGE = (1000, 2000)  # Example range (1000Hz to 2000Hz)

# def apply_frequency_masking(y, sr, freq_range):
#     """
#     Applies frequency masking to the audio signal within the specified frequency range.
#     """
#     # Convert to spectrogram
#     stft = librosa.stft(y)
#     spectrogram = np.abs(stft)

#     # Create frequency mask
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 1)
#     mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
#     spectrogram[mask] = 0

#     # Reconstruct audio signal from masked spectrogram
#     masked_stft = spectrogram * np.exp(1j * np.angle(stft))
#     y_masked = librosa.istft(masked_stft)
#     return y_masked

# def apply_augmentation(y, sr):
#     """
#     Applies frequency masking augmentation and returns the masked audio data with a suffix.
#     """
#     y_masked = apply_frequency_masking(y, sr, MASKING_FREQUENCY_RANGE)
#     return [(y_masked, "frequency_masked")]