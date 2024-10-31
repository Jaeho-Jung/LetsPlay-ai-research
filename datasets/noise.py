import os
from pydub import AudioSegment
import noisereduce as nr
import numpy as np
from scipy.io import wavfile

# directory
_AUDIO_DIR = 'audio/'
_OUTPUT_DIR = 'audio_augmented/'

os.makedirs(_OUTPUT_DIR, exist_ok=True)


def reduce_noise(audio_path, output_path):
    # 오디오 파일을 로드하고 numpy 배열로 변환
    rate, data = wavfile.read(audio_path)

    # remove noise
    reduced_noise = nr.reduce_noise(y=data, sr=rate)

    # save
    wavfile.write(output_path, rate, reduced_noise.astype(np.int16))
    print(f"Saved denoised audio file: {output_path}")


# remove noise (all audio file)
for audio_file in os.listdir(_AUDIO_DIR):
    if not audio_file.endswith('.wav'):
        continue

    # filename
    audio_path = os.path.join(_AUDIO_DIR, audio_file)
    file_name = os.path.splitext(audio_file)[0]
    new_file_name = f"{file_name}_noise.wav"
    output_path = os.path.join(_OUTPUT_DIR, new_file_name)

    # 실행
    reduce_noise(audio_path, output_path)
