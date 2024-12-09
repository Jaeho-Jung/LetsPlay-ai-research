import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("./XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./XTTS-v2/")
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["/content/voice_sample_jh1.wav"])

print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    "오픈AI는 계속해서  o1 기능을 확장할 계획이다. 외부 애플리케이션에서 작업을 수행할 수 있는 함수 호출 기능도 개발 중이다.  오픈AI는 o1에 웹 브라우징 및 광범위한 파일 업로드 처리 기능을 탑재할 계획이다. 챗GPT 프로에는  컴퓨팅 집약적인 생산성 기능도 추가될 예정이다. 최신 버전 o1은 프리뷰 버전보다 더 빠르고 정확한 답변을 제공한다. 이 모델은 어려운 실제 질문에 응답할 때 중대한 오류가 있는 결과물을 생성할 가능성이 34% 감소했다고 테크크런치는 전했다.",
    "ko",
    gpt_cond_latent,
    speaker_embedding
)

wav_chuncks = []
for i, chunk in enumerate(chunks):
    if i == 0:
        print(f"Time to first chunck: {time.time() - t0}")
    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    wav_chuncks.append(chunk)
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)