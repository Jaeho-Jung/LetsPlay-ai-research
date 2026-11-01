# LetsPlay 음성 AI 실험

> 영유아 역할놀이 챗봇 캡스톤에서 진행한 음성 인식 실험과 프로젝트 종료 후의 추론 벤치마크를 기록했습니다.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.45+-FFD21E)](https://huggingface.co/transformers)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)

모델 실험과 서버 코드를 분리했습니다. STT·LLM·TTS를 연결한 백엔드는 [LetsPlay-server](https://github.com/Jaeho-Jung/LetsPlay-server)에 있습니다.

---

## 실험 범위

| 목표 | 설명 |
|---|---|
| 영유아 음성 인식 | 영유아 발화에 맞는 STT 모델을 학습하고 평가 |
| 데이터 증강 | 초기 음성 데이터에 여러 증강 기법을 적용 |
| 온디바이스 추론 | TFLite와 OpenVINO 변환 후 출력과 추론 가능 여부 확인 |
| LLM 파인튜닝 | Unsloth로 LLaMA 3.1 8B 파인튜닝·추론 과정 확인 |
| 프로젝트 종료 후 벤치마크 | 추론 시간과 동시 요청 처리 방식 비교 |

---

## Part 1: 영유아 음성 데이터 증강 및 STT 파인튜닝

### 초기 데이터

초기 데이터는 영유아 음성 200개였습니다. 영유아 발화는 조음이 불완전하고 피치가 높거나 발화가 짧아 인식이 어려울 수 있습니다.

### 데이터 증강

**200개 → 4,200개** (21배 확장)

| 증강 기법 | 설명 |
|---|---|
| SpecAugment & SpecSwap | 스펙트로그램 시간/주파수 대역 마스킹/교체 |
| FilterAugment | 주파수 대역별 진폭 필터링 |
| Mixer (Background Noise) | 일상 소음(숨소리, 청소기, 반려견 등)을 5dB/15dB SNR로 믹싱 |
| Reverberation | 마이크/소원 위치 무작위 시뮬레이션 잔향 효과 |
| Speed Perturbation | 발화 속도 0.9x, 1.1x 조절 |

### Whisper 파인튜닝

증강된 4,200개 데이터셋으로 `openai/whisper-tiny` 파인튜닝을 진행했습니다.

초기 학습에서는 다음과 같은 문제가 있었습니다.
- Training Loss → 0 수렴, Validation Loss 요동
- Dropout 적용 전 WER/CER 0.119048에 고정 수렴

다음 설정으로 다시 학습했습니다.
- Dropout 적용: `attention_dropout=0.2`, `activation_dropout=0.2`
- 정규화: `weight_decay=0.1`
- LR: `1e-5`, Cosine Annealing
- Step 1500에서 증강된 키워드 검증 데이터의 CER은 0.0000이었습니다.
- 소수 키워드(강아지, 고양이 등)에 맞춰진 결과로 판단했습니다. 실제 아동 발화에 대응하려고 AIHub 아동 음성으로 학습된 `whisper-small` 가중치를 사용했습니다.

---

## 온디바이스 추론 검토와 서버 전환

### 시도: 모델 변환 및 런타임 검증

모바일 앱 내부에서 STT 추론을 수행하기 위해 모델 변환 실험을 진행했습니다.

이 단계에서는 **모델 포맷 변환**과 **런타임 비교**를 중심으로 배포 가능성을 확인했습니다. ONNX → TFLite는 포맷 변환이며, OpenVINO INT8 적용은 변환 과정에 양자화가 추가된 경우입니다. 따라서 전체 실험을 일괄적으로 모델 압축 또는 경량화라고 부르지 않습니다.

| 구분 | 접근 | 결과 |
|---|---|---|
| 모델 포맷 변환·런타임 검증 | **ONNX → TFLite** | 변환은 완료했으나 추론 시 문자 수준의 비정상 출력 반복 (e.g., `_88888888Z88ZZZ...`) |
| 모델 변환·INT8 양자화·런타임 검증 | **OpenVINO INT8** | 변환 및 CPU 추론 성공, 런타임 적용 가능성 확인 |

### 한계 및 피벗 (Edge → Cloud)

TFLite의 비정상 출력 원인은 단일 요인으로 확정하지 못했습니다. 변환·추론 과정의 **연산자 지원 및 동적 텐서 처리 문제로 추정**했으며, 프로젝트 기한과 인식 품질을 고려해 **GCP Cloud Run GPU 서버 기반의 클라우드 서버사이드 추론**으로 아키텍처를 전환했습니다.

---

## Part 3: sLLM 파인튜닝 검증 (LLaMA 3.1 8B)

역할놀이에 맞는 페르소나와 문맥 유지를 위한 sLLM 도입을 검토했습니다.

- **모델**: `unsloth/Meta-Llama-3.1-8B` + 4-bit QLoRA + Xformers (Flash Attention 2는 T4 미지원으로 비활성화)
- **환경**: 단일 GPU (T4)
- **결과**: Alpaca 형태 데이터셋 파인튜닝 및 추론 파이프라인 검증 성공

---

## Part 4: [Post-Project] 추론 최적화 및 스케일링 벤치마크

> 캡스톤 프로젝트 완료 후, 실제 서비스 배포를 가정했을 때 발생하는 **초기 모델의 속도 지연**과 **동시 접속 처리** 문제를 해결하기 위한 심화 실험을 진행했습니다.

### 실험 1: 단일 추론 최적화 — CPU vs GPU 비교

**환경**: Google Colab T4 GPU · `elmenwol/whisper-small_aihub_child`

아래 값은 노트북에 남아 있는 **기존 실험 기록**입니다. CPU와 GPU 기록은 워밍업·반복 횟수와 측정 경로가 달라 하나의 통합 벤치마크처럼 직접 비교하지 않습니다.

**CPU 런타임 비교**

동일한 오디오 파일을 대상으로 4개 CPU 스레드, 워밍업 1회 후 5회 실행의 평균을 기록했습니다. 각 함수 호출의 오디오 로드·전처리·생성·디코딩 구간을 포함하지만, 오디오 길이는 기록되지 않았습니다.

| 방법 | 평균 시간 | vs CPU Direct | 프로세스 메모리 |
|---|:---:|:---:|---|
| Direct (FP32) | 16.70s | 1.00x | 2,516 MB (RSS) |
| Dynamic Quantization (INT8) | 65.16s | 0.26x | 3,693 MB (RSS) |
| ONNX Runtime | 12.75s | 1.31x | 5,834 MB (RSS) |

- 이 결과는 **해당 Colab CPU·PyTorch 런타임·동적 양자화 구현 조건에서 INT8 추론 지연이 악화**되었음을 의미합니다. Whisper의 attention 구조 자체가 INT8에 부적합하다는 일반화된 결론은 아닙니다.

**GPU 엔진 비교**

같은 노트북의 동일 오디오 파일, 배치 1에서 각 엔진을 로드한 뒤 워밍업 없이 첫 번째 함수 호출을 측정한 기록입니다. Direct/`torch.compile`과 CTranslate2의 전처리·디코딩 구현 및 생성 옵션이 완전히 같다고 검증하지 않았고, 오디오 길이도 기록되지 않아 수치는 잠정 비교로 봐야 합니다.

| 엔진 | 첫 호출 시간 | 상대값 | 메모리 측정 |
|---|:---:|:---:|---|
| Direct Inference (FP16) | 6.02s | 1.00x | PyTorch allocator 기준 모델 477.8 MB, 피크 549.3 MB |
| `torch.compile` | 0.64s | 9.45x | PyTorch allocator 기준 모델 487.6 MB, 피크 549.2 MB |
| Faster-Whisper (CTranslate2) | 0.80s | 7.53x | **재측정 필요** |

> 기존 9.9MB 값은 `torch.cuda.memory_allocated()`로 수집되어, PyTorch 밖에서 동작하는 CTranslate2의 CUDA 메모리 할당을 반영하지 못했을 가능성이 큽니다. 실제 프로세스 GPU 메모리는 NVML 또는 `nvidia-smi`로 다시 측정해야 합니다. 따라서 9.9MB와 이를 근거로 한 98% 절감 주장은 비교 결과에서 제외했습니다.
>
> 별도 요약에 남아 있는 Direct FP16의 0.39초는 워밍업 후 값으로 기재되어 있으나, 워밍업 횟수·반복 횟수·측정 구간이 충분히 기록되지 않았습니다. 6.02초·0.80초·0.64초와의 직접 비교 및 속도 배수 계산에는 사용하지 않습니다.

> **채택: Faster-Whisper (CTranslate2)**  
> 기록된 지연 시간과 서버 런타임 적용성을 바탕으로 최종 서버에 반영했습니다. 다만 실제 VRAM 우위와 엔진 간 성능 순위는 동일한 오디오·생성 옵션·워밍업·반복 측정 조건에서 재검증해야 합니다.

인스턴스 단가와 과금 기준이 달라질 수 있고, 기존 CPU/GPU 처리시간도 동일 조건의 측정값이 아니므로 비용 배수 결론은 제외했습니다.

### 실험 2: 동시성(Concurrency) 제어 아키텍처 분석

별도의 CTranslate2 FP16 실험에서 동일한 오디오 경로를 반복 사용해 요청 50건에 대한 4가지 처리 전략을 비교했습니다. 모델 로딩과 네트워크 I/O는 측정에서 제외했고, P95는 각 요청 함수의 시작부터 완료까지의 경과 시간으로 계산했습니다. 오디오 길이와 독립적인 워밍업 횟수는 기록되지 않았으며, Baseline은 50건을 순차 실행하고 나머지 전략은 각 구현의 큐·동시 실행 방식을 사용하므로 아래 값은 단일 추론 표와 직접 비교할 수 없습니다.

| 전략 | QPS | P95 지연 | 평가 |
|---|:---:|:---:|---|
| **Baseline (순차)** | 5.28 | **0.259s** | 가장 낮은 지연 |
| num_workers (CT2) | **6.62** | 7.394s | 최고 처리량, 꼬리 지연 심각 |
| Async Queue | 2.18 | 22.016s | 안정성 확보, 속도 감소 |
| Micro-batch | 1.41 | 32.990s | CT2 단일파일 API 한계로 비효율 |

**처리량(QPS) vs 꼬리 지연(Tail Latency) 트레이드오프**
- `num_workers`가 처리량은 25% 높으나 P95 지연이 **28배** 악화 → 실시간 서비스에서 SLA 위반 위험
- `Async Queue`는 속도보다 backpressure와 자원 사용량 제어를 위한 전략이며, N=500+ 부하와 OOM 방지 효과는 후속 검증이 필요

**최종 엔지니어링 의사결정**

> 현재 기록에서는 **Baseline 순차 처리 + Rate Limiting**이 가장 낮은 P95를 보여 최종 서버 전략으로 적용했습니다. 실제 SLA 판단 전에는 오디오 길이, 워밍업, 생성 옵션, 동시 요청 시작 조건과 서버 I/O를 고정한 재측정이 필요합니다. 고가용성 확장 시에는 Async Queue + Redis 기반 수평 확장을 후속안으로 검토합니다.

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
├── 02_model_conversion/                          # Part 2: 모델 변환 & 런타임 배포 가능성 검증
│   ├── onnx_to_tflite.ipynb                       # PyTorch → ONNX → TFLite 변환 파이프라인
│   └── whisper_to_openvino.ipynb                  # OpenVINO INT8 양자화 변환 및 추론
│
├── 03_inference_optimization/                    # Part 4: 추론 최적화 & 동시성 벤치마크
│   ├── gpu_inference_fp16_chunking.ipynb           # GPU 추론 엔진 비교 (Direct/CT2/compile)
│   ├── cpu_inference_benchmark.ipynb               # CPU 추론 벤치마크 (FP32/INT8/ONNX)
│   ├── fastapi_whisper_server.ipynb                # FastAPI 기반 Whisper 서버 프로토타입
│   ├── whisper_inference_gradio.ipynb              # Gradio 인터페이스 추론 데모
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

## 프로젝트에서 바꾼 결정

초기 아동 음성 200개를 증강해 4,200개 학습 데이터를 만들었습니다. Whisper-tiny의 제한된 키워드 검증 결과가 실제 발화에 대한 일반화를 보여주지 못해 아동 음성으로 학습된 Whisper-small 모델로 전환했습니다. TFLite 변환 모델의 출력 문제로 온디바이스 추론을 계속 진행하지 못했고 서버 추론으로 구조를 바꿨습니다.

프로젝트 종료 후에는 같은 오디오 경로의 50건 처리 실험에서 처리량과 P95 지연을 비교했습니다. 이 값에는 네트워크 시간이 포함되지 않아 실제 서비스 응답 시간으로 사용하지 않습니다. 서버 코드는 [LetsPlay-server](https://github.com/Jaeho-Jung/LetsPlay-server)에 있습니다.

---

## Author

**정재호 (Jaeho Jung)** — Team Leader

---

## License

This project is part of the 2024 Capstone Design course (JBNU).
