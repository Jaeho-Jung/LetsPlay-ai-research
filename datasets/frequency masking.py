import os
import librosa
import numpy as np
import soundfile as sf

# directory
_AUDIO_DIR = 'audio/'
_OUTPUT_DIR = 'audio_augmented/'
TARGET_LABEL = '강아지'
MASKING_FREQUENCY_RANGE = (1000, 2000)  # frequency range (ex: 1000Hz~2000Hz)

os.makedirs(_OUTPUT_DIR, exist_ok=True)


def apply_frequency_masking(y, sr, freq_range):
    # spectrogram 변환
    stft = librosa.stft(y)
    spectrogram = np.abs(stft)

    # frequency masking
    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 1)
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    spectrogram[mask] = 0

    # 마스킹된 스펙트로그램을 다시 음성 신호로 변환
    masked_stft = spectrogram * np.exp(1j * np.angle(stft))
    y_masked = librosa.istft(masked_stft)
    return y_masked


# '강아지'만 변환
for audio_file in os.listdir(_AUDIO_DIR):
    if not audio_file.endswith('.wav'):
        continue

    # load audio file
    audio_path = os.path.join(_AUDIO_DIR, audio_file)
    y, sr = librosa.load(audio_path, sr=None)

    # apply frequency masking
    y_masked = apply_frequency_masking(y, sr, MASKING_FREQUENCY_RANGE)

    # file name
    file_name = os.path.splitext(audio_file)[0]
    new_file_name = f"{file_name}_frequency.wav"
    output_path = os.path.join(_OUTPUT_DIR, new_file_name)

    # save frequency-masked audio file
    sf.write(output_path, y_masked, sr)
    print(f"Saved frequency-masked audio file: {output_path}")
