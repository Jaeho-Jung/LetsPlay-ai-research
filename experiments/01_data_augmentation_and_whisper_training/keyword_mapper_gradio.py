import gradio as gr
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

puppy_names = ['강아지', '푸들', '진돗개', '말티즈', '시바견', '비글', '불독', '코카스패니얼', '차우차우', '보더콜리']
cat_names = ['고양이', '러시안블루', '페르시안', '샴고양이', '먼치킨', '스코티시폴드', '노르웨이숲', '뱅갈', '터키시앙고라', '메인쿤']

# 모델과 프로세서 불러오기
checkpoint_path = "kresnik/wav2vec2-large-xlsr-korean"  # 저장된 체크포인트 경로
model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path)
processor = Wav2Vec2Processor.from_pretrained(checkpoint_path)

# 모델 평가 모드로 설정
model.eval()

# 오디오 처리 및 추론 함수
def transcribe(audio_file):

    audio_input, _ = librosa.load(audio_file, sr=16000)

    # 오디오 파일 로드
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values
    
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    for name in puppy_names:
        if name in transcription:
            return '강아지'
    
    for name in cat_names:
        if name in transcription:
            return '고양이'
        
    return transcription

# Gradio 인터페이스 정의
interface = gr.Interface(
    fn=transcribe,  # 추론 함수 연결
    inputs=gr.Audio(type="filepath"),  # 오디오 파일 업로드 입력
    outputs="text",  # 텍스트 출력
    title="Whisper Speech-to-Text",
    description="Upload an audio file, and the Whisper model will transcribe it to text."
)

# Gradio 앱 실행
interface.launch()
