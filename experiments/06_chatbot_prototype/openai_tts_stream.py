import os
import time
from pathlib import Path

from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

speech_file_path = Path(__file__).parent / "speech.mp3"

def main() -> None:
    stream_to_speakers()

def stream_to_speakers() -> None:
    import pyaudio

    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    start_time = time.time()

    with openai.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        response_format="pcm",  # similar to WAV, but without a header chunk at the start.
        input="""
옛날 어느 작은 마을에 "두루미와 여우"라는 별명을 가진 친구 두 명이 살았어요. 두루미는 키가 크고 목이 길어 무엇이든 멀리서도 잘 볼 수 있었고, 여우는 재빠르고 똑똑해서 언제나 새로운 계획을 짜는 걸 좋아했죠.

어느 날, 여우가 두루미에게 말했어요.
"두루미야, 오늘 내가 맛있는 스튜를 끓였는데 같이 먹자!"
두루미는 기뻐하며 여우의 집으로 갔어요. 그런데 스튜는 넓고 얕은 접시에 담겨 있었어요. 두루미의 긴 부리로는 도저히 먹을 수가 없었죠. 여우는 웃으며 스튜를 혼자 다 먹어치웠어요.

다음 날, 두루미가 여우를 자기 집으로 초대했어요. "오늘은 내가 맛있는 수프를 준비했어!"
여우는 신이 나서 두루미의 집에 갔어요. 그런데 이번에는 긴 목이 있는 병에 수프가 담겨 있었어요. 여우는 좁은 입구 때문에 수프를 전혀 먹을 수 없었죠. 두루미는 병에서 맛있게 수프를 마시며 웃었어요.

그 후로 두루미와 여우는 서로를 배려하기로 약속했답니다. 두 사람은 서로의 차이를 이해하며 더욱더 좋은 친구가 되었대요.

이야기의 교훈은, 우리는 각자 다르지만, 그 다름을 이해하고 배려하면 더 나은 관계를 만들 수 있다는 거예요.
""",
    ) as response:
        print(f"Time to first byte: {int((time.time() - start_time) * 1000)}ms")
        for chunk in response.iter_bytes(chunk_size=1024):
            player_stream.write(chunk)

    print(f"Done in {int((time.time() - start_time) * 1000)}ms.")


if __name__ == "__main__":
    main()