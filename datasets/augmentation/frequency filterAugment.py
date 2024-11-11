import numpy as np
from scipy.signal import butter, lfilter

# Filter parameters
FILTER_TYPE = 'low'  # 'low' for low-pass, 'high' for high-pass filter
STEP_CUTOFFS = [500, 1500, 3000]  # Step filter cutoff frequencies

# Apply random filter
def apply_random_filter(y, sr, filter_type='low'):
    # Random cutoff frequency (300Hz - 4000Hz)
    cutoff = np.random.randint(300, 4000)
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(N=4, Wn=normal_cutoff, btype=filter_type, analog=False)
    y_filtered = lfilter(b, a, y)
    return y_filtered, f"random_{cutoff}Hz"

# Apply step filter
def apply_step_filter(y, sr, cutoffs, filter_type='low'):
    y_filtered = y
    for cutoff in cutoffs:
        nyquist = 0.5 * sr
        normal_cutoff = cutoff / nyquist
        b, a = butter(N=4, Wn=normal_cutoff, btype=filter_type, analog=False)
        y_filtered = lfilter(b, a, y_filtered)
    return y_filtered, "step"

# Combined augmentation function for augmentator.py
def apply_augmentation(y, sr):
    augmentations = []

    # Random filter augmentation
    y_random_filtered, random_suffix = apply_random_filter(y, sr, FILTER_TYPE)
    augmentations.append((y_random_filtered, random_suffix))

    # Step filter augmentation
    y_step_filtered, step_suffix = apply_step_filter(y, sr, STEP_CUTOFFS, FILTER_TYPE)
    augmentations.append((y_step_filtered, step_suffix))

    return augmentations