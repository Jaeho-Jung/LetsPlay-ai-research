'''
- The size of the room randomly set uniformly
    - width: between 3 meters to 10 meters
    - length between 3 meters to 8 meters
    - height between 2.5 meters to 6 meters. 
- the target and noise source locations 
    - randomly selected with respect to the microphone. 
    - target source: 
        - azimuth: θ, [−180.0o, 180.0o]
        - elevation: φ, [45.0o, 135.0o]
        - are randomly selected to be in the interval 
- The location of the noise source is also randomly selected, 
    - distribution
        - θ: [−180.0o, 180.0o]
        - φ: [−30.0o, 180.0o]
and , respectively. We intentionally set the
distribution of the noise sources to be wider than that of the target source. 
When the sound source locations (target or nosie) are chosen, we assume that they are at least 0.5 meters away from the wall.
The noise source intensities are set so that the utterance level Signal-to-noise Ratio (SNR) is between 0 dB and
30 dB, with an average of 12 dB over the entire corpus. 
The SNR distribution was also conditioned on the target to mic distance. Fig. 2(a) shows the distribution of the SNR from which
the SNR level of specific utterance is pulled.
- Reverberation of each room: 
    - randomly chosen to be between 0 milliseconds (no reverberation) and 900 milliseconds.
'''

import pyroomacoustics as pra
import soundfile as sf
import random
import numpy as np
import os

from constants import _AUDIO_DIR, _AUDIO_AUG_DIR

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

def reverbate(input_path, output_path):
    audio_data, sample_rate = sf.read(input_path)

    room_width = random.uniform(3, 10)
    room_depth = random.uniform(3, 8)
    room_height = random.uniform(2.5, 6)
    # Room dimensions in meters (example dimensions, modify as needed)
    room_dimensions = [room_width, room_depth, room_height]

    # Create a room with specified dimensions and reverberation time
    reverbation = random.uniform(0, 900)
    # rt60 = desired reverberation time in seconds
    rt60 = 0.5  # Modify this for stronger/weaker reverberation
    room = pra.ShoeBox(room_dimensions, fs=sample_rate, materials=pra.Material(energy_absorption=rt60), max_order=10)

    # Place the sound source in the room at a specific location (modify coordinates as needed)
    source_position = [
        random.uniform(0.5, room_width-0.5),
        random.uniform(0.5, room_depth-0.5),
        random.uniform(0.5, room_height-0.5),
    ]

    target_position = random_position_from_angles(room_dimensions, [-180, 180], [45, 135], source_position)
    room.add_source(target_position, signal=audio_data)

    # Place a microphone at a specific location in the room (modify coordinates as needed)
    mic_position = [
        random.uniform(0.5, room_width-0.5),
        random.uniform(0.5, room_depth-0.5),
        random.uniform(0.5, room_height-0.5),
    ]

    room.add_microphone(mic_position, fs=sample_rate)

    # Simulate the room (this applies the reverb)
    room.simulate()

    # Get the reverberated signal from the microphone
    reverberated_audio = room.mic_array.signals[0]

    # Save the reverberated signal as a new .wav file
    sf.write(output_path, reverberated_audio, sample_rate)

    print("Reverberated audio saved to", output_file)


# Load the original .wav file
for input_file in os.listdir(_AUDIO_DIR):
    if not input_file.endswith('.wav'):
        continue

    input_path = os.path.join(_AUDIO_DIR, input_file)
    file_name = os.path.splitext(input_file)[0]

    for i in range(4):
        output_file = f"{file_name}_reverb_{i}.wav"
        output_path = os.path.join(_AUDIO_AUG_DIR, output_file)

        reverbate(input_path, output_path)