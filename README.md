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
| 1 | 벡터 연산 기초 | Print global indices for 1D vector<br>GPU 벡터 덧셈 (메모리 할당, 호스트-디바이스 전송) | [x] |
| 2 | Add matrix | Matrix 덧셈 cuda , triton                                                       | [ ] |

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
