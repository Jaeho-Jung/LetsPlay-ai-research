import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import random

from constants import _AUDIO_DIR, _AUDIO_AUG_DIR

def load_audio(file_path):
    """Load an audio file."""
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def save_audio(file_path, audio, sr):
    """Save the audio to a .wav file."""
    sf.write(file_path, audio, sr)

def wav_to_log_mel_spectrogram(y, sr, n_mels=40):
    """Convert a waveform to a log-mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def log_mel_spectrogram_to_wav(log_mel_spec, sr, n_iter=32):
    """Convert a log-mel spectrogram back to a waveform."""
    mel_spec = librosa.db_to_power(log_mel_spec)
    y = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_iter=n_iter)
    return y

def swap_freq(spectrogram, max_freq):
    """Swap frequencies in the spectrogram."""
    num_freq_bins, _ = spectrogram.shape

    # Choose Block Size f
    f = random.randint(0, max_freq)

    # Choose f0, f1
    f0 = random.randint(0, num_freq_bins - 2*f-1)
    f1 = random.randint(f0+f, num_freq_bins - f-1)

    # Swap the selected regions
    temp = spectrogram[f0:f0+f, :].copy()
    spectrogram[f0:f0+f, :] = spectrogram[f1:f1+f, :]
    spectrogram[f1:f1+f, :] = temp

    return spectrogram

def swap_time(spectrogram, max_time):
    """Swap time in the spectrogram."""
    _, num_time_steps = spectrogram.shape

    # Choose Block Size t
    t = random.randint(0, max_time)

    # Choose t0, t1
    t0 = random.randint(0, num_time_steps - 2*t-1)
    t1 = random.randint(t0+t, num_time_steps - t-1)

    # Swap the selected regions
    temp = spectrogram[:, t0:t0+t].copy()
    spectrogram[:, t0:t0+t] = spectrogram[:, t1:t1+t]
    spectrogram[:, t1:t1+t] = temp

    return spectrogram

def specswap(spectrogram, max_freq=7, max_time=10):
    """Perform SpecSwap on the spectrogram."""
    swapped_spectrogram = spectrogram.copy()
    num_freq_bins, num_time_steps = swapped_spectrogram.shape

    max_time = min(max_time, num_time_steps)
    max_freq = min(max_freq, num_freq_bins)

    swapped_spectrogram = swap_freq(swapped_spectrogram, max_freq)
    swapped_spectrogram = swap_time(swapped_spectrogram, max_time)

    return swapped_spectrogram

def plot_spectrogram(spectrogram, sr, title="Spectrogram"):
    """Plot the spectrogram."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), 
                             sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def specswap_augment(input_path, output_path):
    # Load audio
    audio, sr = load_audio(input_path)

    # Convert audio to spectrogram
    spectrogram = wav_to_log_mel_spectrogram(audio, sr)

    # # Plot original spectrogram
    # plot_spectrogram(spectrogram, title="Original Spectrogram")

    # Perform SpecSwap augmentation
    augmented_spectrogram = specswap(spectrogram)

    # # Plot augmented spectrogram
    # plot_spectrogram(augmented_spectrogram, sr, title="Augmented Spectrogram (SpecSwap)")

    # Convert augmented spectrogram back to audio
    augmented_audio = log_mel_spectrogram_to_wav(augmented_spectrogram, sr)

    # Save augmented audio to .wav file
    save_audio(output_path, augmented_audio, sr)

    print(f"Augmented audio saved to {output_path}")

for input_file in os.listdir(_AUDIO_DIR):
    if not input_file.endswith('.wav'):
        continue

    input_path = os.path.join(_AUDIO_DIR, input_file)
    file_name = os.path.splitext(input_file)[0]
    output_file = f"{file_name}_specswap.wav"
    output_path = os.path.join(_AUDIO_AUG_DIR, output_file)

    specswap_augment(input_path, output_path)