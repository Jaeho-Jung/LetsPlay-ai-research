import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

# directory
_AUDIO_DIR = 'audio/'
_OUTPUT_DIR = 'audio_augmented/'
FILTER_TYPE = 'low'  # 'low' for low-pass, 'high' for high-pass filter
STEP_CUTOFFS = [500, 1500, 3000]  # 스텝 필터의 차단 주파수 단계들

os.makedirs(_OUTPUT_DIR, exist_ok=True)

# apply random filter
def apply_random_filter(y, sr, filter_type='low'):
    # random 차단 주파수 cutoff 설정
    cutoff = np.random.randint(300, 4000)  # 300Hz ~ 4000Hz random value
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(N=4, Wn=normal_cutoff, btype=filter_type, analog=False)
    y_filtered = lfilter(b, a, y)
    return y_filtered, cutoff


#apply step filter
def apply_step_filter(y, sr, cutoffs, filter_type='low'):
    y_filtered = y
    for cutoff in cutoffs:
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        b, a = butter(N=4, Wn=normal_cutoff, btype=filter_type, analog=False)
        y_filtered = lfilter(b, a, y_filtered)
    return y_filtered


for audio_file in os.listdir(_AUDIO_DIR):
    if not audio_file.endswith('.wav'):
        continue

    # audio file load
    audio_path = os.path.join(_AUDIO_DIR, audio_file)
    y, sr = librosa.load(audio_path, sr=None)

    # apply random filter
    y_random_filtered, random_cutoff = apply_random_filter(y, sr, FILTER_TYPE)
    random_filtered_name = f"{os.path.splitext(audio_file)[0]}_filterAugment_random_{random_cutoff}.wav"
    random_output_path = os.path.join(_OUTPUT_DIR, random_filtered_name)
    sf.write(random_output_path, y_random_filtered, sr)
    print(f"Saved random-filtered audio file: {random_output_path}")

    # apply step filter
    y_step_filtered = apply_step_filter(y, sr, STEP_CUTOFFS, FILTER_TYPE)
    step_filtered_name = f"{os.path.splitext(audio_file)[0]}_filterAugment_step.wav"
    step_output_path = os.path.join(_OUTPUT_DIR, step_filtered_name)
    sf.write(step_output_path, y_step_filtered, sr)
    print(f"Saved step-filtered audio file: {step_output_path}")
