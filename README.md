# LetsPlay — 영유아 언어 발달 실시간 음성 역할놀이 챗봇

> 기업 연계 캡스톤 프로젝트 — **영유아 맞춤형 실시간 음성 인식 및 역할놀이 챗봇** 서버 개발을 위한 AI R&D 및 추론 최적화 실험

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.45+-FFD21E)](https://huggingface.co/transformers)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)

**Note**: 본 레포지토리는 **모델링 실험, 추론 최적화, 트러블슈팅 과정**을 담고 있습니다.  
이 실험 결과를 바탕으로 최종 구현된 서비스(앱/백엔드) 코드는 **[LetsPlay-server](https://github.com/Jaeho-Jung/LetsPlay-server)**에서 확인하실 수 있습니다.

---

## 1. 프로젝트 개요 (Project Overview)

**영유아 언어 발달을 돕는 실시간 음성 상호작용 역할놀이 챗봇 서버**를 개발하며, 시스템 아키텍처 설계, AI 모델 파이프라인(STT–LLM–TTS) 통합, 클라우드 인프라 배포, 추론 속도 최적화까지 전 과정을 수행했습니다.

### 담당 역할

| 역할 | 내용 |
|---|---|
| **시스템 아키텍처** | 클라이언트 → WebSocket → STT → LLM → TTS → 클라이언트 파이프라인 설계 |
| **AI 파이프라인 통합** | Faster-Whisper(STT), GPT-4o-mini(LLM), OpenAI TTS 스트리밍 연결 |
| **클라우드 배포** | Docker + GCP Cloud Run GPU(T4) 기반 서버 배포 |
| **추론 최적화** | CPU/GPU 벤치마크, CTranslate2 엔진 도입, 동시성 전략 수립 |

### 기술 스택

| Category | Technologies |
|---|---|
| **Backend** | Python, FastAPI, WebSockets |
| **STT** | Faster-Whisper (CTranslate2), HuggingFace Whisper |
| **LLM** | OpenAI GPT-4o-mini (Streaming) |
| **TTS** | OpenAI TTS-1 (Streaming), Coqui XTTS v2 |
| **Deep Learning** | PyTorch, TensorFlow, Unsloth (QLoRA) |
| **Optimization** | ONNX, TFLite, OpenVINO, torch.compile |
| **Infra** | Docker, GCP Cloud Run, ngrok |

---

## 2. 아키텍처 및 시스템 설계 (Architecture & System Design)

### 전체 파이프라인

```
┌──────────┐   WebSocket   ┌───────────────────────────────────────────────┐   WebSocket   ┌──────────┐
│  Client  │ ────────────→ │  ① STT (Faster-Whisper)  [Blocking]          │ ←──────────── │  Client  │
│  (App)   │               │  ② LLM (GPT-4o-mini)     [Streaming] ──┐    │               │  (App)   │
│          │ ←──────────── │  ③ TTS (OpenAI TTS-1)    [Streaming] ←─┘    │ ────────────→ │          │
└──────────┘   Audio PCM   └───────────────────────────────────────────────┘   Audio PCM   └──────────┘
```

### 설계 의사결정 (Trade-off)

| 결정 | 근거 |
|---|---|
| **STT: Blocking 처리** | 영유아 발화는 짧고 문맥 의존적 → 부분 전사(Streaming STT)보다 전체 발화 완료 후 한 번에 전사하는 것이 정확도 우위 |
| **LLM+TTS: Streaming** | 체감 대기시간(TTFB) 최소화 → LLM 토큰 생성 즉시 TTS로 전달, 약 0.5s 내 첫 음성 재생 |
| **Edge → Cloud 피벗** | 온디바이스 TFLite/OpenVINO 변환 시 추론 품질 저하 확인 → 실시간성과 정확도를 위해 GCP Cloud Run GPU 서버로 전환 |

---

## 3. 성능 최적화: 추론 엔진 벤치마킹 및 도입 (Inference Optimization)

> 프로젝트 진행 당시 FP16+Chunking 기반 GPU 추론을 적용했으나, 서비스 확장을 대비하여 추론 속도와 리소스 효율을 극대화하기 위한 후속 벤치마킹을 진행했습니다.

### 3-1. CPU vs GPU 벤치마크 (Colab T4)

| 방법 | 시간 | vs CPU Direct | 메모리 |
|---|:---:|:---:|---|
| **CPU** Direct (FP32) | 16.70s | 1.00x | 2,516 MB (RSS) |
| **CPU** Dynamic Quant (INT8) | 65.16s | 0.26x | 3,693 MB (RSS) |
| **CPU** ONNX Runtime | 12.75s | 1.31x | 5,834 MB (RSS) |
| **GPU** Direct (FP16+SDPA) | 0.39s | **42.8x** | 549 MB (VRAM) |
| **GPU** Faster-Whisper (CT2) | 0.80s | **20.9x** | 9.9 MB (VRAM) |
| **GPU** torch.compile | 0.64s | **26.1x** | 549 MB (VRAM) |

**핵심 결과:**
- GPU 전환만으로 CPU 대비 **최대 42.8배** 속도 향상
- 비용 효율: GCP 기준 시간당 인스턴스 비용은 GPU가 높으나, 처리 속도 차이로 요청당 비용은 **GPU가 21배 저렴**

### 3-2. GPU 추론 엔진 비교 및 최종 결정

| 엔진 | 속도 | 모델 VRAM | 피크 VRAM | 특징 |
|---|:---:|:---:|:---:|---|
| Direct Inference (FP16) | 1.00x (6.02s) | 477.8 MB | 549.3 MB | Baseline |
| **torch.compile** | **9.45x** (0.64s) | 487.6 MB | 549.2 MB | 최고 속도, 높은 VRAM |
| **Faster-Whisper** | 7.53x (0.80s) | **9.9 MB** | **9.9 MB** | VRAM 98% 절감 |

> **최종 채택: Faster-Whisper (CTranslate2)**  
> `torch.compile`이 최고 속도를 기록했으나, Cloud Run 환경의 제한된 VRAM(T4/L4)을 효율적으로 사용하기 위해 메모리 점유율을 **98% 절감**(9.9MB)하면서도 실시간 기준을 충족(0.80s)하는 Faster-Whisper를 채택하여 실제 서버 코드에 반영.

---

## 4. 동시성 제어 및 스케일링 전략 (Concurrency & Scaling)

> 단일 GPU 환경에서 다중 사용자 요청(N=50) 시 발생하는 병목 현상을 분석하고 최적 전략을 수립했습니다.

### 벤치마크 결과 (N=50, Colab T4)

| 전략 | QPS | Total | P95 | 평가 |
|---|:---:|:---:|:---:|---|
| **Baseline (순차)** | 5.28 | 9.48s | **0.259s** | 가장 낮은 지연 |
| num_workers (CT2) | **6.62** | 7.55s | 7.394s | 최고 처리량, 꼬리 지연 심각 |
| Async Queue | 2.18 | 22.88s | 22.016s | 안정성 확보, 속도 감소 |
| Micro-batch | 1.41 | 35.37s | 32.990s | CT2 단일파일 API 한계로 비효율 |

### 아키텍처 전략

| 환경 | 전략 | 근거 |
|---|---|---|
| **현재: 단일 GPU** | Baseline(순차) + Rate Limiting | CTranslate2 내부 커널 스케줄링이 이미 최적, 외부 큐는 오버헤드만 추가. P95=0.259s로 가장 안정적 |
| **향후: 고가용성 (N>100)** | Async Queue + Multi-Worker | OOM 방지, `QueueFullError` → HTTP 429 graceful degradation |
| **향후: 다중 GPU** | Redis Queue + GPU당 Worker | 처리량 선형 확장, 수평 스케일아웃 |

---

## 5. R&D 실험 상세

### R&D 1: 음성 데이터 증강 (Data Augmentation)

수집 가능한 초기 데이터(200개)의 부족 문제를 해결하기 위한 증강 파이프라인을 구축했습니다.

**200개 → 4,200개** (21배 확장)

| 증강 기법 | 설명 |
|---|---|
| SpecAugment & SpecSwap | 스펙트로그램 시간/주파수 대역 마스킹/교체 |
| FilterAugment | 주파수 대역별 진폭 필터링 |
| Mixer (Background Noise) | 일상 소음(숨소리, 청소기, 반려견 등)을 5dB/15dB SNR로 믹싱 |
| Reverberation | 마이크/소원 위치 무작위 시뮬레이션 잔향 효과 |
| Speed Perturbation | 발화 속도 0.9x, 1.1x 조절 |

### R&D 2: Whisper 모델 파인튜닝 및 트러블슈팅

증강된 4,200개 데이터셋으로 `openai/whisper-tiny` 파인튜닝을 진행하며 과적합 문제를 분석했습니다.

**Issue: 지표 수렴 및 과적합**
- Training Loss → 0 수렴, Validation Loss 요동
- WER/CER이 0.119048에 고정 수렴

**Troubleshooting**
- Dropout 적용: `attention_dropout=0.2`, `activation_dropout=0.2`
- 정규화: `weight_decay=0.01`
- LR 스케줄링: `1e-5` → `1e-6` → `5e-5`, Cosine Annealing
- **결론**: 모델 크기(tiny)의 한계 및 단어 위주 학습셋 특성상 한계 확인 → `whisper-small` 스케일업 근거

### R&D 3: 온디바이스 TFLite & OpenVINO 경량화

모바일 기기 내부에서 STT 추론을 수행하기 위한 모델 변환 실험을 진행했습니다.

| 접근 | 결과 |
|---|---|
| **ONNX → TFLite** | TFLite 변환 성공, 추론 시 무의미한 토큰 반복 출력 |
| **OpenVINO INT8** | OpenVINO 변환 및 CPU 추론 성공, 프레임워크 다각도 검토 수행 |

**한계 및 배움**: 변환 과정 정보 손실 및 연산자 호환성 문제 → 클라우드 서버사이드 추론(Cloud Run) 기반 아키텍처 피벗의 계기

### R&D 4: Unsloth 기반 sLLM 파인튜닝

역할놀이 챗봇의 두뇌(Brain)를 만들기 위해 로컬 GPU 환경에서 효율적 파인튜닝을 테스트했습니다.

- **모델**: `unsloth/Meta-Llama-3.1-8B` + 4-bit QLoRA + Flash Attention
- **환경**: 단일 GPU (T4)
- **결과**: Alpaca 형태 데이터셋 파인튜닝 및 추론 파이프라인 검증 성공 ✅

---

## 6. 프로젝트 구조

```
experiments/
├── 01_data_augmentation_and_whisper_training/   # R&D 1+2: 데이터 증강 & Whisper 파인튜닝
│   ├── whisper_finetune_train.ipynb              # Whisper-tiny 파인튜닝 메인 실험
│   ├── wav2vec2_comparison.ipynb                  # Wav2Vec2 아키텍처 비교 검토
│   ├── whisper_model_download.ipynb               # 학습된 모델 다운로드/저장
│   └── keyword_mapper_gradio.py                   # 키워드 매핑 Gradio 데모
│
├── 02_model_conversion/                          # R&D 3: TFLite & OpenVINO 경량화 시도
│   ├── onnx_to_tflite.ipynb                       # PyTorch → ONNX → TFLite 변환 파이프라인
│   └── whisper_to_openvino.ipynb                  # OpenVINO INT8 양자화 변환 및 추론
│
├── 03_inference_optimization/                    # 추론 최적화 & 동시성 벤치마크
│   ├── gpu_inference_fp16_chunking.ipynb           # GPU 추론 엔진 비교 (Direct/CT2/compile)
│   ├── cpu_inference_benchmark.ipynb               # CPU 추론 벤치마크 (FP32/INT8/ONNX)
│   ├── benchmark_concurrency.py                    # 동시성 벤치마크 스크립트
│   ├── whisper_service_queue.py                    # Async Queue & Micro-batch 서비스
│   ├── benchmark_analysis.md                       # 동시성 벤치마크 분석 보고서
│   ├── whisper_inference_gradio.ipynb              # Gradio 기반 추론 데모
│   └── fastapi_whisper_server.ipynb                # FastAPI + ngrok 서버 프로토타입
│
├── 04_sllm_finetuning/                           # R&D 4: LLaMA 파인튜닝
│   ├── unsloth_llama3_finetune.ipynb              # Unsloth + LoRA + Flash Attention
│   ├── llama3_qlora_finetune.py                   # QLoRA 4-bit 양자화 파인튜닝
│   └── dataset_converter.py                       # 배민 데이터셋 JSON→CSV 변환기
│
├── 05_tts_experiments/                           # TTS 엔진 비교 실험
│   ├── coqui_tts_exploration.ipynb                # Coqui TTS 모델 탐색
│   ├── xtts_v2_voice_cloning.ipynb                # XTTS v2 음성 클로닝 실험
│   ├── xtts_streaming_inference.py                # XTTS 스트리밍 추론
│   └── elevenlabs_tts_api.py                      # ElevenLabs API TTS 실험
│
└── 06_chatbot_prototype/                         # GPT 역할놀이 챗봇 프로토타입
    ├── gpt_roleplay_chat.py                       # GPT 역할놀이 채팅
    ├── gpt_roleplay_stream.py                     # GPT 스트리밍 채팅
    └── openai_tts_stream.py                       # OpenAI TTS 스트리밍
```

---

## 7. 회고 및 배운 점 (Lessons Learned)

| 교훈 | 상세 |
|---|---|
| **유연한 아키텍처 피벗** | 온디바이스(Edge) 환경의 하드웨어 제약을 마주했을 때, 클라우드(GCP)로 신속하게 전환하여 실시간성과 정확도를 모두 확보 |
| **수치 기반 의사결정** | "무조건 빠른 것"이 정답이 아니라, **인프라 비용(VRAM·GCP 요금) × 사용자 경험(Latency) × 운영 안정성(OOM 방지)**의 트레이드오프를 벤치마크 데이터로 분석하여 최적 엔진(Faster-Whisper) 선택 |
| **데이터 파이프라인의 중요성** | 부족한 초기 데이터(200개)를 21배 증강하는 파이프라인 구축, 도메인 맞춤 데이터 가공 역량 습득 |
| **실용주의적 접근** | 경량화/파인튜닝의 현실적 한계를 빠르게 인정하고, AIHub 오픈소스 가중치(`whisper-small`) 도입으로 유연한 의사결정 |

**최종 서비스 코드**: [LetsPlay-server](https://github.com/Jaeho-Jung/LetsPlay-server)

---

## Author

**정재호 (Jaeho Jung)** — Team Leader

---

## License

This project is part of the 2024 Capstone Design course (JBNU).