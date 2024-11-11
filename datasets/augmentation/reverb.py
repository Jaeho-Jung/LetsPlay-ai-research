import pyroomacoustics as pra
import numpy as np
import random

def random_position_from_angles(room_dimensions, azimuth_range, elevation_range, mic_position, distance=1.0):
    azimuth = np.deg2rad(np.random.uniform(*azimuth_range))
    elevation = np.deg2rad(np.random.uniform(*elevation_range))
    x = mic_position[0] + distance * np.cos(elevation) * np.cos(azimuth)
    y = mic_position[1] + distance * np.cos(elevation) * np.sin(azimuth)
    z = mic_position[2] + distance * np.sin(elevation)
    # Ensure source is at least 0.5m from any wall
    x = min(max(0.5, x), room_dimensions[0] - 0.5)
    y = min(max(0.5, y), room_dimensions[1] - 0.5)
    z = min(max(0.5, z), room_dimensions[2] - 0.5)
    return [x, y, z]

def apply_reverberation(y, sr):
    """
    Applies reverberation effect to the audio data.
    """
    room_width = random.uniform(3, 10)
    room_depth = random.uniform(3, 8)
    room_height = random.uniform(2.5, 6)
    room_dimensions = [room_width, room_depth, room_height]

    # Set reverberation time (between 0 and 900 ms)
    reverb_time = random.uniform(0, 900) / 1000  # Convert ms to seconds
    room = pra.ShoeBox(room_dimensions, fs=sr, materials=pra.Material(energy_absorption=reverb_time), max_order=10)

    # Set a random target position in the room
    mic_position = [
        random.uniform(0.5, room_width - 0.5),
        random.uniform(0.5, room_depth - 0.5),
        random.uniform(0.5, room_height - 0.5)
    ]
    target_position = random_position_from_angles(room_dimensions, [-180, 180], [45, 135], mic_position)

    room.add_source(target_position, signal=y)
    room.add_microphone(mic_position, fs=sr)

    # Simulate the room to apply reverberation
    room.simulate()

    # Get the reverberated signal from the microphone
    reverberated_audio = room.mic_array.signals[0]
    return reverberated_audio

def apply_augmentation(y, sr):
    """
    Applies reverberation augmentation multiple times and returns the augmented audio data with suffixes.
    """
    augmentations = []
    for i in range(4):  # Apply the augmentation 4 times with varying reverberation
        y_reverberated = apply_reverberation(y, sr)
        suffix = f"reverb_{i}"
        augmentations.append((y_reverberated, suffix))

    return augmentations
