import os
import sys
import importlib.util
import librosa
import soundfile as sf


# Parse arguments from the command line
if len(sys.argv) < 3:
    print("Usage: python augmentator.py <audio_dir> <output_dir>")
    sys.exit(1)

AUDIO_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load each augmentation script from the augmentation directory
AUGMENTATION_DIR = 'augmentation'
augmentation_scripts = [
    os.path.join(AUGMENTATION_DIR, f) for f in os.listdir(AUGMENTATION_DIR)
    if f.endswith('.py')
]

def load_and_apply_augmentation(script_path, audio_data, sample_rate, output_name_prefix):
    """
    Dynamically loads and applies an augmentation function from a given script.
    """
    spec = importlib.util.spec_from_file_location("module.name", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Apply augmentation if the script has an `apply_augmentation` function
    if hasattr(module, "apply_augmentation"):
        augmentations = module.apply_augmentation(audio_data, sample_rate)
        for augmented_data, suffix in augmentations:
            output_path = os.path.join(OUTPUT_DIR, f"{output_name_prefix}_{suffix}.wav")
            sf.write(output_path, augmented_data, sample_rate)
            # print(f"Saved augmented audio file: {output_path}")

for audio_file in os.listdir(AUDIO_DIR):
    if not audio_file.endswith('.wav'):
        continue

    audio_path = os.path.join(AUDIO_DIR, audio_file)
    y, sr = librosa.load(audio_path, sr=None)
    output_name_prefix = os.path.splitext(audio_file)[0]

    for script in augmentation_scripts:
        load_and_apply_augmentation(script, y, sr, output_name_prefix)
