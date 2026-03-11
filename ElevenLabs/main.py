
import os
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)


def text_to_speech_file(text: str) -> str:
    # Calling the text_to_speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id="uyVNoMrnUku1dZyVEXwD", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5", # use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # uncomment the line below to play the audio back
    # play(response)

    # Generating a unique file name for the output MP3 file
    save_file_path = f"{uuid.uuid4()}.mp3"

    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Return the path of the saved audio file
    return save_file_path

text = """
오, 고양이가 정말 매력적인 동물이지 않니? 그들은 무척 조용히 움직이고, 높은 곳에서도 쉽게 뛰어오르곤 해. 고양이는 또한 촉감이 섬세해서 어두운 곳에서도 길을 잘 찾을 수 있단다. 네가 만약 고양이와 함께 모험을 떠난다면, 어떤 곳에 가고 싶니? 예를 들어, 어떤 멋진 사원이나 숲을 탐험해보는 건 어떨까?
"""
text_to_speech_file(text)