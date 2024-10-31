import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf

# directory
_AUDIO_DIR = 'audio/'
_OUTPUT_DIR = 'audio_augmented/'
MASKING_FREQUENCY_BANDWIDTH = 20  # masking frequency bandwidth

os.makedirs(_OUTPUT_DIR, exist_ok=True)


def apply_spec_augment_frequency_masking(y, sr, freq_bandwidth):
    # spectrogram
    stft = librosa.stft(y)
    spectrogram = np.abs(stft)

    # frequency masking
    num_freq_bins = spectrogram.shape[0]
    mask_start = np.random.randint(0, num_freq_bins - freq_bandwidth)
    spectrogram[mask_start:mask_start + freq_bandwidth, :] = 0

    # 마스킹된 스펙트로그램을 다시 음성 신호로 변환
    masked_stft = spectrogram * np.exp(1j * np.angle(stft))
    y_masked = librosa.istft(masked_stft)
    return y_masked


for audio_file in os.listdir(_AUDIO_DIR):
    if not audio_file.endswith('.wav'):
        continue

    # audio file load
    audio_path = os.path.join(_AUDIO_DIR, audio_file)
    y, sr = librosa.load(audio_path, sr=None)

    # Apply SpecAugment Frequency Masking
    y_masked = apply_spec_augment_frequency_masking(y, sr, MASKING_FREQUENCY_BANDWIDTH)

    # new file name
    file_name = os.path.splitext(audio_file)[0]
    new_file_name = f"{file_name}_frequency.wav"
    output_path = os.path.join(_OUTPUT_DIR, new_file_name)

    # save
    sf.write(output_path, y_masked, sr)
    print(f"Saved frequency-masked audio file: {output_path}")
