import os
from pydub import AudioSegment

# directory
_AUDIO_DIR = 'audio/'
_OUTPUT_DIR = 'audio_augmented/'
SPEEDS = [0.8, 1.2]  # speed list

os.makedirs(_OUTPUT_DIR, exist_ok=True)


def change_speed(audio_segment, speed=1.0):
    # speed
    return audio_segment._spawn(audio_segment.raw_data, overrides={
        "frame_rate": int(audio_segment.frame_rate * speed)
    }).set_frame_rate(audio_segment.frame_rate)


for audio_file in os.listdir(_AUDIO_DIR):
    if not audio_file.endswith('.wav'):
        continue

    file_name = os.path.splitext(audio_file)[0]
    audio_path = os.path.join(_AUDIO_DIR, audio_file)

    # audio file loading
    audio = AudioSegment.from_wav(audio_path)

    # save files
    for speed in SPEEDS:
        modified_audio = change_speed(audio, speed=speed)

        # file name setting
        new_file_name = f"{file_name}_{speed}.wav"
        new_file_path = os.path.join(_OUTPUT_DIR, new_file_name)

        # save audio files
        modified_audio.export(new_file_path, format="wav")
        print(f"Saved augmented audio file: {new_file_name}")


