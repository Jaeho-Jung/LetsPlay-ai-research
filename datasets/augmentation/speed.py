from pydub import AudioSegment
import numpy as np


# Speed change values
SPEEDS = [0.8, 0.9, 1.1, 1.2]  # Example speed factors

def change_speed(audio_segment, speed=1.0):
    """
    Changes the speed of an audio segment.
    """
    modified_segment = audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": int(audio_segment.frame_rate * speed)
    }).set_frame_rate(audio_segment.frame_rate)

    # Adjust volume to match the original segment's dBFS
    return modified_segment.apply_gain(audio_segment.dBFS - modified_segment.dBFS)

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


def apply_augmentation(y, sr):
    """
    Applies speed adjustment augmentation at different rates and returns augmented audio data with suffixes.
    """
    # Convert numpy audio array to pydub AudioSegment for processing
    audio_segment = numpy_to_audiosegment(y, sr)

    augmentations = []
    for speed in SPEEDS:
        # Apply speed change
        modified_audio = change_speed(audio_segment, speed=speed)
        
       # Convert modified audio back to numpy format and normalize to [-1, 1] if needed
        modified_audio_data = np.array(modified_audio.get_array_of_samples(), dtype=np.float32)
        modified_audio_data /= np.max(np.abs(modified_audio_data))  # Normalize to [-1, 1]
        
        suffix = f"speed_{speed}"

        augmentations.append((modified_audio_data, suffix))

    return augmentations

import sounddevice as sd
import numpy as np

def play_audio_from_np_array(audio_data, sample_rate):
    """
    Plays audio from a numpy array.

    Args:
        audio_data (np.ndarray): The audio data as a numpy array.
        sample_rate (int): The sample rate of the audio data.
    """
    # Play the audio array
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()  # Wait until playback is finished


import librosa

y, sr = librosa.load('/Users/jaeho/Workspace/Capstone2024/Capstone2024_CodeX/datasets/audio_test/audio_0_강아지_test.wav', sr=None)