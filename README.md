# GPU 100 Days Challenge

100일 동안 CUDA kernel과 Triton kernel을 연습하는 프로젝트입니다.

## 📋 목차

- [설치 방법](#설치-방법)
- [사용법](#사용법)
- [프로젝트 구조](#프로젝트-구조)
- [100일 커리큘럼](#100일-커리큘럼)
- [참고 자료](#참고-자료)

## 🚀 설치 방법

### 필수 요구사항

- Python >= 3.11
- CUDA Toolkit (12.6 이상 권장)
- PyTorch 2.7.0
- CMake >= 3.18 (CMake 빌드 사용 시)

### 설치

1. **저장소 클론**

```bash
git clone <repository-url>
cd gpu-100days
```

2. **가상환경 생성 및 활성화**

```bash
# uv 사용 (권장)
uv venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows

# 또는 venv 사용
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

3. **의존성 설치 및 빌드**

```bash
# uv 사용
uv sync --no-build-isolation

# 또는 pip 사용
pip install -e .
```

빌드가 완료되면 CUDA 확장 모듈(`cuda_ops`)이 설치됩니다.

### CMake를 사용한 직접 빌드 (선택사항)

```bash
cd csrc
mkdir build && cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
make
```

## 💻 사용법

### 기본 사용 예제

```python
from gpu_100days import vector_add
import torch

# CUDA 텐서 생성
a = torch.randn(1000, device='cuda', dtype=torch.float32)
b = torch.randn(1000, device='cuda', dtype=torch.float32)

# CUDA 커널을 사용한 벡터 덧셈
c = vector_add(a, b)

# PyTorch 기본 연산과 비교
c_pytorch = a + b
print(torch.allclose(c, c_pytorch))  # True
```

### 예제 실행

```bash
python src/gpu_100days/vector_add.py
```

## 📁 프로젝트 구조

```
gpu-100days/
├── csrc/                    # CUDA 소스 파일 (.cu)
│   ├── CMakeLists.txt       # CMake 빌드 설정
│   └── vectorAdd.cu         # 예제 CUDA 커널
├── src/
│   ├── cuda_ops.pyi         # 타입 스텁 파일
│   └── gpu_100days/         # Python 패키지
│       ├── __init__.py
│       └── vector_add.py    # Python 래퍼
├── setup.py                 # setuptools 빌드 스크립트
├── pyproject.toml          # 프로젝트 설정
└── README.md
```

## 📚 100일 커리큘럼

이 커리큘럼은 [100DaysForGPU](https://github.com/mino-park7/100DaysForGPU) 레포지토리를 참고하여 구성되었습니다.

### Week 1-2: CUDA 기초 (Day 1-14)

| Day | 주제 | 내용 | 상태 |
|-----|------|------|------|
| 1 | 벡터 연산 기초 | Print global indices for 1D vector<br>GPU 벡터 덧셈 (메모리 할당, 호스트-디바이스 전송) | [ ] |
| 2 | Device 함수 | `__device__` 함수 사용<br>스레드별 계산 | [ ] |
| 3 | 2D 행렬 연산 | 2D 행렬 덧셈<br>행/열 인덱스를 스레드에 매핑<br>커스텀 함수를 사용한 행렬 변환 | [ ] |
| 4 | Layer Normalization | Shared memory 사용<br>평균/분산 계산 | [ ] |
| 5 | Reduction 연산 | 병렬 벡터 합계<br>Shared memory 최적화 | [ ] |
| 6 | 고급 기법 | SM ID 조회 (PTX 사용)<br>Shared memory를 사용한 Softmax<br>행렬 전치<br>Python-CUDA 통합 | [ ] |
| 7 | 행렬 곱셈 | Naive 구현<br>Tiled matmul with shared memory<br>1D convolution | [ ] |
| 8 | 다양한 커널 | Matrix-vector multiplication<br>이미지 처리 (RGB to grayscale, blur)<br>Self-attention with online softmax | [ ] |
| 9 | Flash Attention 기초 | Minimal Flash Attention 구현<br>Shared memory tiling | [ ] |
| 10 | Flash Attention 통합 | PyTorch 확장 바인딩<br>테스트 및 벤치마크 | [ ] |
| 11 | Backward pass | Gradient 계산<br>Custom CUDA kernel과 PyTorch 비교 | [ ] |
| 12 | Softmax 최적화 | Shared memory를 사용한 Softmax<br>Tiled kernel 구현 | [ ] |
| 13 | RMS Normalization | Naive 구현 (V1)<br>Warp-reduce 최적화 (V2)<br>Float4 사용 | [ ] |
| 14 | Flash Attention 2 | Forward/backward pass<br>2D convolution | [ ] |

### Week 3-4: 고급 CUDA 기법 (Day 15-28)

| Day | 주제 | 내용 | 상태 |
|-----|------|------|------|
| 15 | Attention 커널 | Single-headed attention<br>Batched/tiled dot-product<br>Sparse matrix multiplication (CSR) | [ ] |
| 16 | Attention backward | Gradient computation<br>Forward & backward passes | [ ] |
| 17 | cuBLAS 활용 | Dot products, axpy<br>Max/min 연산<br>기타 BLAS 연산 | [ ] |
| 18 | Warp 및 Atomic 연산 | Warp-based reduction<br>Custom atomic operations | [ ] |
| 19 | cuBLAS Matrix Multiplication | cuBLAS를 사용한 행렬 곱셈<br>Self-attention 예제 | [ ] |
| 20 | RoPE | RoPE 커널 구현<br>PyTorch 확장 및 벤치마크 | [ ] |
| 21 | Memory Coalescing | 메모리 접근 패턴 최적화<br>Coalesced memory access | [ ] |
| 22 | Bank Conflicts | Shared memory bank conflicts 해결<br>최적화 기법 | [ ] |
| 23 | Occupancy 최적화 | 스레드 블록 구성 최적화<br>리소스 사용량 분석 | [ ] |
| 24 | 고급 최적화 실습 | 종합 최적화 기법 적용 | [ ] |
| 25 | 복합 커널 구현 | 여러 커널 통합<br>파이프라인 구성 | [ ] |
| 26 | 성능 프로파일링 | Nsight Compute/Systems 사용<br>병목 지점 분석 | [ ] |
| 27 | 벤치마킹 | 성능 측정 및 비교<br>최적화 결과 검증 | [ ] |
| 28 | Week 3-4 종합 프로젝트 | 지금까지 학습한 내용 종합 실습 | [ ] |

### Week 5-8: Triton 기초 (Day 29-56)

| Day | 주제 | 내용 | 상태 |
|-----|------|------|------|
| 29 | Triton 설치 및 환경 설정 | Triton 설치<br>환경 구성<br>기본 예제 실행 | [ ] |
| 30 | Triton 기본 커널 | 기본 커널 작성<br>Python과의 통합<br>간단한 벡터 연산 | [ ] |
| 31 | Triton 벡터 연산 | 벡터 덧셈/곱셈<br>Element-wise operations | [ ] |
| 32 | Triton 행렬 연산 | 행렬 곱셈 기초<br>Tile 기반 접근 | [ ] |
| 33 | Triton Reduction | Reduction 연산 구현<br>다양한 reduction 패턴 | [ ] |
| 34 | Triton 기초 연산 심화 | 복합 연산<br>조건부 연산 | [ ] |
| 35 | Triton 기초 종합 | Week 5 종합 실습 | [ ] |
| 36 | Block-level 프로그래밍 | Tile 기반 연산 설계<br>Block 구조 이해 | [ ] |
| 37 | Shared Memory 활용 | Triton에서의 메모리 최적화<br>Shared memory 패턴 | [ ] |
| 38 | Block-level 최적화 | 고급 tile 기법<br>메모리 계층 활용 | [ ] |
| 39 | Atomic Operations | Atomic 연산 구현<br>Race condition 해결 | [ ] |
| 40 | Multi-stage Reduction | 다단계 reduction<br>효율적인 reduction 패턴 | [ ] |
| 41 | Fused Kernels | 여러 연산 통합<br>Fusion 최적화 | [ ] |
| 42 | Triton 최적화 종합 | Week 6 종합 실습 | [ ] |
| 43 | Flash Attention in Triton | Triton으로 Flash Attention 구현 | [ ] |
| 44 | Fused Attention Kernels | Attention 커널 fusion<br>성능 최적화 | [ ] |
| 45 | Attention 최적화 | 다양한 attention 패턴<br>최적화 기법 | [ ] |
| 46 | Layer Normalization in Triton | Triton으로 LayerNorm 구현 | [ ] |
| 47 | Activation Functions | 다양한 activation 함수<br>Triton 구현 | [ ] |
| 48 | Custom Operations | 커스텀 연산 구현<br>재사용 가능한 패턴 | [ ] |
| 49 | Triton 실전 프로젝트 1 | Attention 기반 모듈 구현 | [ ] |
| 50 | Triton 실전 프로젝트 2 | Normalization 및 Activation 통합 | [ ] |
| 51 | Triton 실전 프로젝트 3 | 복합 커널 구현 | [ ] |
| 52 | Triton 실전 프로젝트 4 | 성능 최적화 및 벤치마킹 | [ ] |
| 53 | Triton 실전 프로젝트 5 | CUDA와 Triton 비교 | [ ] |
| 54 | Triton 실전 프로젝트 6 | 실전 적용 사례 | [ ] |
| 55 | Triton 실전 프로젝트 7 | 종합 프로젝트 | [ ] |
| 56 | Week 7-8 종합 | Triton 학습 내용 정리 및 복습 | [ ] |

### Week 9-12: 고급 주제 (Day 57-84)

| Day | 주제 | 내용 | 상태 |
|-----|------|------|------|
| 57 | Profiling Tools | Nsight Compute 심화<br>성능 분석 기법 | [ ] |
| 58 | Memory Bandwidth 최적화 | 메모리 대역폭 분석<br>최적화 전략 | [ ] |
| 59 | Compute 최적화 | 연산 최적화<br>Instruction-level 최적화 | [ ] |
| 60 | 성능 최적화 실습 1 | 실제 커널 최적화 | [ ] |
| 61 | 성능 최적화 실습 2 | 벤치마크 및 비교 | [ ] |
| 62 | 성능 최적화 실습 3 | 프로파일링 기반 최적화 | [ ] |
| 63 | 성능 최적화 종합 | Week 9 종합 프로젝트 | [ ] |
| 64 | Multi-kernel Fusion | 여러 커널 통합<br>Fusion 전략 | [ ] |
| 65 | Pipeline Optimization | 파이프라인 최적화<br>비동기 실행 | [ ] |
| 66 | Custom Autograd Functions | PyTorch autograd 통합<br>Custom backward 구현 | [ ] |
| 67 | 복합 커널 실습 1 | Fusion 커널 구현 | [ ] |
| 68 | 복합 커널 실습 2 | Pipeline 최적화 적용 | [ ] |
| 69 | 복합 커널 실습 3 | Autograd 통합 | [ ] |
| 70 | 복합 커널 종합 | Week 10 종합 프로젝트 | [ ] |
| 71 | Custom Attention Layers | Attention 레이어 구현<br>다양한 attention 메커니즘 | [ ] |
| 72 | Feed-forward Networks | FFN 구현<br>Activation 통합 | [ ] |
| 73 | Residual Connections | Residual connection 구현<br>Skip connection 패턴 | [ ] |
| 74 | Transformer Block 구현 | 완전한 Transformer block<br>모든 컴포넌트 통합 | [ ] |
| 75 | Transformer 실습 1 | 작은 모델 구현 | [ ] |
| 76 | Transformer 실습 2 | 중간 규모 모델 | [ ] |
| 77 | Transformer 실습 3 | 최적화 및 벤치마킹 | [ ] |
| 78 | 성능 비교 | CUDA vs Triton 성능 비교<br>최적 커널 선택 | [ ] |
| 79 | 메모리 사용량 분석 | 메모리 프로파일링<br>최적화 전략 | [ ] |
| 80 | 배포 최적화 | 프로덕션 환경 최적화<br>배포 전략 | [ ] |
| 81 | 최적화 실습 1 | 성능 튜닝 | [ ] |
| 82 | 최적화 실습 2 | 메모리 최적화 | [ ] |
| 83 | 최적화 실습 3 | 배포 준비 | [ ] |
| 84 | Week 11-12 종합 | 고급 주제 종합 프로젝트 | [ ] |

### Week 13-14: 마무리 (Day 85-100)

| Day | 주제 | 내용 | 상태 |
|-----|------|------|------|
| 85 | End-to-end 모델 설계 | 전체 모델 아키텍처 설계<br>컴포넌트 계획 | [ ] |
| 86 | 모델 구현 1 | 핵심 컴포넌트 구현 | [ ] |
| 87 | 모델 구현 2 | 보조 컴포넌트 구현 | [ ] |
| 88 | Training Loop 통합 | 학습 루프 구현<br>최적화 및 검증 | [ ] |
| 89 | 에러 처리 | Robust error handling<br>예외 상황 대응 | [ ] |
| 90 | 문서화 | 코드 문서화<br>API 문서 작성 | [ ] |
| 91 | 테스트 작성 | Unit test<br>Integration test | [ ] |
| 92 | 프로덕션 준비 | 배포 체크리스트<br>최종 검증 | [ ] |
| 93 | 자유 주제 프로젝트 1 | 원하는 커널/모델 선택 및 설계 | [ ] |
| 94 | 자유 주제 프로젝트 2 | 구현 및 최적화 | [ ] |
| 95 | 자유 주제 프로젝트 3 | 창의적 최적화 적용 | [ ] |
| 96 | 자유 주제 프로젝트 4 | 최종 완성 및 검증 | [ ] |
| 97 | 100일 회고 1 | 학습 내용 정리<br>주요 성과 요약 | [ ] |
| 98 | 100일 회고 2 | 어려웠던 점 및 해결 과정 | [ ] |
| 99 | 포트폴리오 작성 1 | 프로젝트 정리<br>문서화 | [ ] |
| 100 | 포트폴리오 작성 2 | 최종 정리 및 공유 준비 | [ ] |

## 📖 참고 자료

### 공식 문서
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Triton Documentation](https://triton-lang.org/)

### 참고 레포지토리
- [100DaysForGPU](https://github.com/mino-park7/100DaysForGPU) - 이 프로젝트의 참고 레포지토리
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Triton Examples](https://github.com/openai/triton)

### 학습 자료
- [CUDA by Example](https://developer.nvidia.com/cuda-example)
- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0)

## 🛠️ 개발 팁

### CUDA 커널을 PyTorch에 통합하는 방법

1. **커널 작성**: `csrc/` 디렉토리에 `.cu` 파일 작성
2. **바인딩 작성**: PyTorch 텐서를 받는 래퍼 함수 작성
3. **Python 래퍼**: `src/gpu_100days/`에 Python 인터페이스 작성
4. **빌드**: `pip install -e .` 또는 `uv sync`로 빌드

### 디버깅

- `compile_commands.json`이 자동으로 생성되어 IDE에서 코드 완성 지원
- CUDA 에러 체크: `cudaGetLastError()` 사용
- PyTorch 텐서 검증: `TORCH_CHECK` 매크로 사용

### 성능 프로파일링

```bash
# Nsight Compute 사용
ncu --set full python your_script.py

# Nsight Systems 사용
nsys profile python your_script.py
```
