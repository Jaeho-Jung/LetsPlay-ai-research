# import numpy as np
# import librosa
# import random

# def wav_to_log_mel_spectrogram(y, sr, n_mels=40):
#     """Convert a waveform to a log-mel spectrogram."""
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
#     log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
#     return log_mel_spec

# def log_mel_spectrogram_to_wav(log_mel_spec, sr, n_iter=32):
#     """Convert a log-mel spectrogram back to a waveform."""
#     mel_spec = librosa.db_to_power(log_mel_spec)
#     y = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_iter=n_iter)
#     return y

# def swap_freq(spectrogram, max_freq):
#     """Swap frequencies in the spectrogram."""
#     num_freq_bins, _ = spectrogram.shape
#     f = random.randint(0, max_freq)
#     f0 = random.randint(0, num_freq_bins - 2*f-1)
#     f1 = random.randint(f0+f, num_freq_bins - f-1)
#     temp = spectrogram[f0:f0+f, :].copy()
#     spectrogram[f0:f0+f, :] = spectrogram[f1:f1+f, :]
#     spectrogram[f1:f1+f, :] = temp
#     return spectrogram

# def swap_time(spectrogram, max_time):
#     """Swap time in the spectrogram."""
#     _, num_time_steps = spectrogram.shape
#     t = random.randint(0, max_time)
#     t0 = random.randint(0, num_time_steps - 2*t-1)
#     t1 = random.randint(t0+t, num_time_steps - t-1)
#     temp = spectrogram[:, t0:t0+t].copy()
#     spectrogram[:, t0:t0+t] = spectrogram[:, t1:t1+t]
#     spectrogram[:, t1:t1+t] = temp
#     return spectrogram

# def specswap(spectrogram, max_freq=4, max_time=7):
#     """Perform SpecSwap on the spectrogram."""
#     swapped_spectrogram = spectrogram.copy()
#     num_freq_bins, num_time_steps = swapped_spectrogram.shape
#     max_time = min(max_time, num_time_steps)
#     max_freq = min(max_freq, num_freq_bins)
#     swapped_spectrogram = swap_freq(swapped_spectrogram, max_freq)
#     swapped_spectrogram = swap_time(swapped_spectrogram, max_time)
#     return swapped_spectrogram

# def apply_augmentation(y, sr):
#     """
#     Applies SpecSwap augmentation to the audio and returns the augmented audio data with a suffix.
#     """
#     # Convert audio to spectrogram
#     spectrogram = wav_to_log_mel_spectrogram(y, sr)
#     # Perform SpecSwap augmentation
#     augmented_spectrogram = specswap(spectrogram)
#     # Convert augmented spectrogram back to audio
#     augmented_audio = log_mel_spectrogram_to_wav(augmented_spectrogram, sr)
#     return [(augmented_audio, "specswap")]
