from pydub import AudioSegment
import numpy as np

# Speed change values
SPEEDS = [0.8, 1.2]  # Example speed factors

def change_speed(audio_segment, speed=1.0):
    """
    Changes the speed of an audio segment.
    """
    return audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": int(audio_segment.frame_rate * speed)
    }).set_frame_rate(audio_segment.frame_rate)

def apply_augmentation(y, sr):
    """
    Applies speed adjustment augmentation at different rates and returns augmented audio data with suffixes.
    """
    # Convert numpy audio array to pydub AudioSegment for processing
    audio_segment = AudioSegment(
        y.tobytes(), 
        frame_rate=sr, 
        sample_width=y.dtype.itemsize, 
        channels=1
    )

    augmentations = []
    for speed in SPEEDS:
        # Apply speed change
        modified_audio = change_speed(audio_segment, speed=speed)
        
        # Convert modified audio back to numpy format
        modified_audio_data = np.array(modified_audio.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
        suffix = f"speed_{speed}"

        augmentations.append((modified_audio_data, suffix))

    return augmentations