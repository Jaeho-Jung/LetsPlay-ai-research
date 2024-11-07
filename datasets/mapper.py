import os
import csv
import sys
from constants import _TRANSCRIPT_OUTPUT_FILE

def create_mapping(audio_dirs):
    mapping = []
    for audio_dir in audio_dirs:
        for audio_file in os.listdir(audio_dir):
            if not audio_file.endswith('.wav'):
                continue
            
            file_name = os.path.splitext(audio_file)[0]
            file_name_list = file_name.split('_')

            transcription = file_name_list[2]  # Adjust this as needed based on file naming convention
            audio_path = os.path.join(audio_dir, audio_file)
            mapping.append({'file': audio_path, 'text': transcription})
    
    return mapping

def write_to_csv(mapping, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in mapping:
            writer.writerow(item)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mapper.py <audio_dir> <audio_aug_dir>")
        sys.exit(1)

    audio_dir = sys.argv[1]
    audio_aug_dir = sys.argv[2]
    audio_dirs = [audio_dir, audio_aug_dir]

    mapping = create_mapping(audio_dirs)
    write_to_csv(mapping, _TRANSCRIPT_OUTPUT_FILE)
