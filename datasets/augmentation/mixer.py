from pydub import AudioSegment
import numpy as np
import os

# Volume ratio for mixing background noise
VOLUME_RATIO = 0.5  # Example volume ratio
_BACKGROUND_DIR = 'background_noise/background_noise_9'

def mix_with_background(main_audio, background_audio, volume_ratio=0.5):
    """
    Mixes the main audio with the background audio at a specified volume ratio.
    """
    # Control background volume relative to main audio
    adjusted_background = background_audio - (background_audio.dBFS - main_audio.dBFS + volume_ratio)
    # Mix the audio
    mixed_audio = main_audio.overlay(adjusted_background)
    return mixed_audio

def apply_augmentation(y, sr, background_dir=_BACKGROUND_DIR):
    """
    Loads background files from `background_dir` and applies background mixing augmentation.
    Returns a list of mixed audio data with suffixes for each background file.
    """
    # Convert the input audio (numpy array) into pydub format
    main_audio = AudioSegment(
        y.tobytes(), 
        frame_rate=sr, 
        sample_width=y.dtype.itemsize, 
        channels=1
    )

    augmentations = []
    for background_file in os.listdir(background_dir):
        if not background_file.endswith('.wav'):
            continue
        
        # Load background audio
        background_path = os.path.join(background_dir, background_file)
        background_audio = AudioSegment.from_file(background_path)

        # Apply background mixing
        mixed_audio = mix_with_background(main_audio, background_audio, VOLUME_RATIO)

        # Convert mixed audio back to numpy format for output
        mixed_audio_data = np.array(mixed_audio.get_array_of_samples(), dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
        suffix = f"mixer_{os.path.splitext(background_file)[0]}"
        
        augmentations.append((mixed_audio_data, suffix))

    return augmentations
