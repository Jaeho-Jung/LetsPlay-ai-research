# LetsPlay — 영유아 역할놀이 챗봇 AI R&D

> 기업 연계 캡스톤 프로젝트 — **영유아 맞춤형 실시간 음성 인식 및 역할놀이 챗봇** 구현을 위한 AI 사전 연구, 데이터 실험, 추론 최적화 벤치마크 기록

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.45+-FFD21E)](https://huggingface.co/transformers)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)

이 레포지토리는 영유아 역할놀이 챗봇(LetsPlay) 구현을 위한 **AI 모델 사전 연구**, **데이터 실험**, 그리고 프로젝트 종료 후 수행한 **추론 최적화 벤치마크**를 담고 있습니다.  
이 연구 결과를 반영하여 구축된 실제 서비스 백엔드는 [LetsPlay-server](https://github.com/Jaeho-Jung/LetsPlay-server)에서 확인하실 수 있습니다.

---

## 연구 목표

| 목표 | 설명 |
|---|---|
| **영유아 발화 특화 STT** | 불완전한 조음과 높은 피치의 영유아 음성을 정확히 인식하는 모델 연구 |
| **데이터 파이프라인** | 수집이 어려운 영유아 음성 데이터를 대체/보완하는 다양한 오디오 증강 기법 구현 |
| **온디바이스 AI (Edge AI)** | 엣지 환경에서의 실시간 추론을 위한 모델 경량화(TFLite/OpenVINO) 및 한계 검토 |
| **sLLM 파인튜닝** | LLaMA 3.1 8B 모델을 Unsloth로 파인튜닝하는 파이프라인 검증 |
| **[Post-Project] 추론 최적화** | 프로덕션 배포를 가정한 추론 속도·비용·동시성 처리 심화 벤치마크 |

---

## Part 1: 영유아 음성 데이터 증강 및 STT 파인튜닝

### 문제 정의

수집 가능한 초기 데이터가 200개에 불과하고, 영유아 발화는 불완전한 조음, 높은 피치, 짧은 발화라는 특수성을 가져 일반 STT 모델로는 인식률이 낮습니다.

### 해결: 데이터 증강 파이프라인

**200개 → 4,200개** (21배 확장)

| 증강 기법 | 설명 |
|---|---|
| SpecAugment & SpecSwap | 스펙트로그램 시간/주파수 대역 마스킹/교체 |
| FilterAugment | 주파수 대역별 진폭 필터링 |
| Mixer (Background Noise) | 일상 소음(숨소리, 청소기, 반려견 등)을 5dB/15dB SNR로 믹싱 |
| Reverberation | 마이크/소원 위치 무작위 시뮬레이션 잔향 효과 |
| Speed Perturbation | 발화 속도 0.9x, 1.1x 조절 |

### Whisper 파인튜닝 및 트러블슈팅

증강된 4,200개 데이터셋으로 `openai/whisper-tiny` 파인튜닝을 진행했습니다.

**Issue: 지표 수렴 및 과적합**
- Training Loss → 0 수렴, Validation Loss 요동
- WER/CER이 0.119048에 고정 수렴

**Troubleshooting**
- Dropout 적용: `attention_dropout=0.2`, `activation_dropout=0.2`
- 정규화: `weight_decay=0.01`
- LR 스케줄링: `1e-5` → `1e-6` → `5e-5`, Cosine Annealing
- **결론**: 모델 크기(tiny)의 한계 확인 → `whisper-small` 스케일업 및 AIHub 오픈소스 가중치 도입 결정

---

## Part 2: 온디바이스(Edge AI) 경량화 실험 및 아키텍처 피벗

### 시도: 모바일 오프라인 추론

모바일 앱 내부에서 STT 추론을 수행하기 위해 모델 변환 실험을 진행했습니다.

| 접근 | 결과 |
|---|---|
| **ONNX → TFLite** | 변환 성공, 추론 시 무의미한 토큰 반복 출력 (`[50258, 50264... 918, 918]`) |
| **OpenVINO INT8** | 변환 및 CPU 추론 성공, 프레임워크 다각도 검토 수행 |

### 한계 및 피벗 (Edge → Cloud)

연산자 미지원 및 동적 텐서 할당 이슈로 인한 비정상 출력 토큰 발생. 프로젝트 기한 및 인식 품질 확보를 위해 **GCP Cloud Run GPU 서버 기반의 클라우드 서버사이드 추론**으로 아키텍처를 전환했습니다.

---

## Part 3: sLLM 파인튜닝 검증 (LLaMA 3.1 8B)

역할놀이에 맞는 페르소나와 문맥 유지를 위한 sLLM 도입을 검토했습니다.

- **모델**: `unsloth/Meta-Llama-3.1-8B` + 4-bit QLoRA + Flash Attention
- **환경**: 단일 GPU (T4)
- **결과**: Alpaca 형태 데이터셋 파인튜닝 및 추론 파이프라인 검증 성공

---

## Part 4: [Post-Project] 추론 최적화 및 스케일링 벤치마크

> 캡스톤 프로젝트 완료 후, 실제 서비스 배포를 가정했을 때 발생하는 **초기 모델의 속도 지연**과 **동시 접속 처리** 문제를 해결하기 위한 심화 실험을 진행했습니다.

### 실험 1: 단일 추론 최적화 — CPU vs GPU 비교

**환경**: Google Colab T4 GPU · `elmenwol/whisper-small_aihub_child`

| 방법 | 시간 | vs CPU Direct | 메모리 |
|---|:---:|:---:|---|
| **CPU** Direct (FP32) | 16.70s | 1.00x | 2,516 MB (RSS) |
| **CPU** Dynamic Quant (INT8) | 65.16s | 0.26x | 3,693 MB (RSS) |
| **CPU** ONNX Runtime | 12.75s | 1.31x | 5,834 MB (RSS) |
| **GPU** Direct (FP16+SDPA) | 0.39s | **42.8x** | 549 MB (VRAM) |
| **GPU** Faster-Whisper (CT2) | 0.80s | **20.9x** | 9.9 MB (VRAM) |
| **GPU** torch.compile | 0.64s | **26.1x** | 549 MB (VRAM) |

- CPU INT8 동적 양자화는 Whisper attention 구조에 비적합하여 **역효과** — CPU 최적화 상한은 1.3x에 불과
- GCP Cloud Run 기준 요청당 비용: GPU가 단가는 높으나 처리량 차이로 **21배 저렴**

**GPU 엔진 비교 및 최종 채택**

| 엔진 | 속도 | 모델 VRAM | 피크 VRAM |
|---|:---:|:---:|:---:|
| Direct Inference (FP16) | 1.00x (6.02s) | 477.8 MB | 549.3 MB |
| torch.compile | **9.45x** (0.64s) | 487.6 MB | 549.2 MB |
| **Faster-Whisper (CT2)** | 7.53x (0.80s) | **9.9 MB** | **9.9 MB** |

> **채택: Faster-Whisper (CTranslate2)**  
> `torch.compile`이 최고 속도이나, Cloud Run의 제한된 VRAM 환경에서 메모리를 **98% 절감**(9.9MB)하면서도 실시간 기준(0.80s)을 충족하는 Faster-Whisper를 최종 서버에 반영.

### 실험 2: 동시성(Concurrency) 제어 아키텍처 분석

N=50 동시 요청 시나리오에서 4가지 전략을 비교했습니다.

| 전략 | QPS | P95 지연 | 평가 |
|---|:---:|:---:|---|
| **Baseline (순차)** | 5.28 | **0.259s** | 가장 낮은 지연 |
| num_workers (CT2) | **6.62** | 7.394s | 최고 처리량, 꼬리 지연 심각 |
| Async Queue | 2.18 | 22.016s | 안정성 확보, 속도 감소 |
| Micro-batch | 1.41 | 32.990s | CT2 단일파일 API 한계로 비효율 |

**처리량(QPS) vs 꼬리 지연(Tail Latency) 트레이드오프**
- `num_workers`가 처리량은 25% 높으나 P95 지연이 **28배** 악화 → 실시간 서비스에서 SLA 위반 위험
- `Async Queue`의 가치는 속도가 아닌 운영 안정성(OOM 방지, N=500+ 환경)

**최종 엔지니어링 의사결정**

> 단일 GPU에서는 CTranslate2의 내부 커널 스케줄링이 이미 최적화되어 있으므로, 외부 큐 없이 **Baseline 순차 처리 + Rate Limiting** 방식을 최종 서버에 적용. 고가용성 확장 시에는 Async Queue + Redis 기반 수평 확장 로드맵 수립.

---

## 프로젝트 구조

```
experiments/
├── 01_data_augmentation_and_whisper_training/   # Part 1: 데이터 증강 & Whisper 파인튜닝
│   ├── whisper_finetune_train.ipynb              # Whisper-tiny 파인튜닝 메인 실험
│   ├── wav2vec2_comparison.ipynb                  # Wav2Vec2 아키텍처 비교 검토
│   ├── whisper_model_download.ipynb               # 학습된 모델 다운로드/저장
│   └── keyword_mapper_gradio.py                   # 키워드 매핑 Gradio 데모
│
├── 02_model_conversion/                          # Part 2: TFLite & OpenVINO 경량화 시도
│   ├── onnx_to_tflite.ipynb                       # PyTorch → ONNX → TFLite 변환 파이프라인
│   └── whisper_to_openvino.ipynb                  # OpenVINO INT8 양자화 변환 및 추론
│
├── 03_inference_optimization/                    # Part 4: 추론 최적화 & 동시성 벤치마크
│   ├── gpu_inference_fp16_chunking.ipynb           # GPU 추론 엔진 비교 (Direct/CT2/compile)
│   ├── cpu_inference_benchmark.ipynb               # CPU 추론 벤치마크 (FP32/INT8/ONNX)
│   ├── benchmark_concurrency.py                    # 동시성 벤치마크 스크립트
│   ├── whisper_service_queue.py                    # Async Queue & Micro-batch 서비스 구현
│   └── benchmark_analysis.md                       # 동시성 벤치마크 분석 보고서
│
├── 04_sllm_finetuning/                           # Part 3: LLaMA 3.1 파인튜닝
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
    ├── gpt_roleplay_chat.py
    ├── gpt_roleplay_stream.py
    └── openai_tts_stream.py
```

---

## 결론 및 회고 (Lessons Learned)

데이터를 가공하는 모델링 영역부터 인프라의 한계를 극복하는 추론 최적화까지, **AI 프로덕트의 전체 사이클**을 경험했습니다.

| 교훈 | 상세 |
|---|---|
| **수치 기반 의사결정** | "무조건 빠른 것"이 정답이 아님. VRAM 비용 × Latency × 운영 안정성의 트레이드오프를 벤치마크 데이터로 분석하여 최적 엔진 선택 |
| **유연한 아키텍처 피벗** | 온디바이스 한계를 마주했을 때 클라우드로 신속 전환, 기술적 결정의 논리적 근거를 수립하는 역량 습득 |
| **데이터 파이프라인** | 초기 데이터 200개를 21배 증강하는 파이프라인 구축, 도메인 맞춤 데이터 가공 역량 습득 |
| **실용주의적 접근** | 경량화/파인튜닝의 현실적 한계를 빠르게 인정하고, 오픈소스(`whisper-small`) 도입으로 유연한 의사결정 |

**최종 서비스 코드**: [LetsPlay-server](https://github.com/Jaeho-Jung/LetsPlay-server)

---

## Author

**정재호 (Jaeho Jung)** — Team Leader

---

## License

This project is part of the 2024 Capstone Design course (JBNU).