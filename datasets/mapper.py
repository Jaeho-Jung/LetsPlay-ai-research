import os
import csv
import sys
from constants import _AUDIO_DIR, _AUDIO_AUG_DIR, _TRANSCRIPT_OUTPUT_FILE

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
    if len(sys.argv) < 4:
        audio_dirs = [_AUDIO_DIR, _AUDIO_AUG_DIR]
        output_file = _TRANSCRIPT_OUTPUT_FILE
        print(f"No arguments provided.")
        exit
    else:
        audio_dirs = [sys.argv[1], sys.argv[2]]
        output_file = sys.argv[3]
    
    mapping = create_mapping(audio_dirs)
    write_to_csv(mapping, output_file)
