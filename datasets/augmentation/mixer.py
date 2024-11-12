from pydub import AudioSegment
import numpy as np
import os

# Volume ratio for mixing background noise
_TARGET_SNR_DB = 5  # Example volume ratio
_BACKGROUND_DIR = 'background_noise/background_noise_9'

from pydub import AudioSegment
import math

def mix_audio_with_snr(main_audio, background_audio, target_snr_db=_TARGET_SNR_DB):
    """
    Mixes main audio with background noise at a specified SNR.

    Args:
        main_audio (AudioSegment): Main audio segment.
        background_audio (AudioSegment): Background noise segment.
        target_snr_db (float): Desired Signal-to-Noise Ratio in dB.

    Returns:
        AudioSegment: Mixed audio segment.
    """
    # Calculate the RMS of both audio segments
    main_rms = main_audio.rms
    background_rms = background_audio.rms

    # Calculate the required background audio gain to achieve the target SNR
    snr_ratio = 10 ** (target_snr_db / 20)  # Convert dB to linear ratio
    desired_background_rms = main_rms / snr_ratio
    adjustment_db = 20 * math.log10(desired_background_rms / background_rms)

    # Adjust background volume and overlay it with main audio
    adjusted_background = background_audio.apply_gain(adjustment_db)
    return main_audio.overlay(adjusted_background)

def numpy_to_audiosegment(y, sr):
    """
    Converts a numpy array to a pydub AudioSegment object, handling conversions to 16-bit PCM.
    
    Args:
        y (numpy.ndarray): Audio data as a numpy array.
        sr (int): Sample rate of the audio.

    Returns:
        AudioSegment: A pydub AudioSegment object representing the audio.
    """
    # If the data is float, convert it to int16
    if y.dtype in [np.float32, np.float64]:
        y = np.int16(y * 32767)  # Scale to 16-bit PCM range

    # Ensure the data is in byte format for AudioSegment
    audio_data = y.tobytes()

    # Create the AudioSegment with the appropriate settings
    audio_segment = AudioSegment(
        data=audio_data,
        sample_width=2,  # 16-bit audio (2 bytes per sample)
        frame_rate=sr,
        channels=1  # Adjust if the audio is stereo
    )
    
    return audio_segment

def apply_augmentation(y, sr, background_dir=_BACKGROUND_DIR):
    """
    Loads background files from `background_dir` and applies background mixing augmentation.
    Returns a list of mixed audio data with suffixes for each background file.
    """
    # Convert the input audio (numpy array) into pydub format
    main_audio = numpy_to_audiosegment(y, sr)

    augmentations = []
    for background_file in os.listdir(background_dir):
        if not background_file.endswith('.wav'):
            continue
        
        # Load background audio
        background_path = os.path.join(background_dir, background_file)
        background_audio = AudioSegment.from_file(background_path)

        for target_snr_db in [10]:
            # Apply background mixing
            mixed_audio = mix_audio_with_snr(main_audio, background_audio, target_snr_db)

            # Convert mixed audio back to numpy format for output
            mixed_audio_data = np.array(mixed_audio.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
            suffix = f"mixer_{os.path.splitext(background_file)[0]}"
            
            augmentations.append((mixed_audio_data, suffix))

    return augmentations
