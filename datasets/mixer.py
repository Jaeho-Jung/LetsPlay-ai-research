from pydub import AudioSegment
import os

# directory
_AUDIO_DIR = 'audio/'
_BACKGROUND_DIR = 'background_noise/background_noise_9/'
_OUTPUT_DIR = 'audio_augmented/'
VOLUME_RATIO = 0.5  # volume ratio

# output directory
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def mix_with_background(audio_file_path, background_file_path, output_file_path, volume_ratio=0.5):
    # main audio load
    main_audio = AudioSegment.from_file(audio_file_path)

    # background audio load and control volume
    background_audio = AudioSegment.from_file(background_file_path)
    background_audio = background_audio - (background_audio.dBFS - main_audio.dBFS + volume_ratio)

    # mixed audio
    mixed_audio = main_audio.overlay(background_audio)

    # save
    mixed_audio.export(output_file_path, format="wav")
    print(f"Saved mixed audio file: {output_file_path}")


# mix and save
for audio_file in os.listdir(_AUDIO_DIR):
    if not audio_file.endswith('.wav'):
        continue

    for background_file in os.listdir(_BACKGROUND_DIR):
        if not background_file.endswith('.wav'):
            continue

        # filename
        background_path = os.path.join(_BACKGROUND_DIR, background_file)
        file_name = os.path.splitext(audio_file)[0]
        background_name = os.path.splitext(background_file)[0]
        output_file = os.path.join(_OUTPUT_DIR, f"{file_name}_Mixer_{background_name}.wav")

        # apply Mixer and save files
        mix_with_background(os.path.join(_AUDIO_DIR, audio_file), background_path, output_file, VOLUME_RATIO)
