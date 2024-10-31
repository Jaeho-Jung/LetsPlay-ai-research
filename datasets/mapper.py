import os
import csv

from words import WORDS
from constants import _AUDIO_DIR, _AUDIO_AUG_DIR, _TRANSCRIPT_OUTPUT_FILE

# Directories
_AUDIO_DIRS = [_AUDIO_DIR, _AUDIO_AUG_DIR]

mapping = []

for _AUDIO_DIR in _AUDIO_DIRS:
    for audio_file in os.listdir(_AUDIO_DIR):
        if not audio_file.endswith('.wav'):
            continue
        
        
        file_name = os.path.splitext(audio_file)[0]
        file_name_list = file_name.split('_')

        # words = WORDS.values()
        # transcription = [word for word in words if word in file_name_list]
        transcription = file_name_list[2]    
        
        audio_path = os.path.join(_AUDIO_DIR, audio_file)
        mapping.append({'file': audio_path, 'text': transcription})

    # Write to CSV
    with open(_TRANSCRIPT_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in mapping:
            writer.writerow(item)

