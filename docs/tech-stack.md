# 기술 스택과 선택 이유

## api 컨테이너 (`Dockerfile.api`)

베이스 이미지: `python:${PYTHON_VERSION}-slim` + PyTorch wheel (`--index-url`로 CPU/CUDA 분기)

> [hub.docker.com/_/python/tags](https://hub.docker.com/_/python/tags)

`python:3.12-slim`을 베이스로 쓰고 torch/torchvision을 PyTorch 공식 wheel index에서 설치함. macOS(CPU)와 Linux(CUDA) 모두에서 동일한 Dockerfile을 사용할 수 있음. `.env`의 `TORCH_INDEX`로 CPU/CUDA wheel을 전환함.

> **Linux 전용 대안**: macOS 지원이 불필요하면 `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`을 베이스로 쓰는 게 더 간단함. torch가 이미 포함되어 있어 별도 설치 불필요. [hub.docker.com/r/pytorch/pytorch/tags](https://hub.docker.com/r/pytorch/pytorch/tags) 참고

| 패키지 | 역할 | 선택 이유 |
|--------|------|-----------|
| **FastAPI** | 모델 서빙 API | 비동기 지원, 자동 API 문서 생성, Pydantic 타입 검증 |
| **Uvicorn** | ASGI 서버 | FastAPI 기본 서버, 비동기 요청 처리 |
| **Pydantic** | 데이터 검증 | FastAPI와 통합, API 입출력 스키마 정의 |
| **python-multipart** | 파일 업로드 파싱 | FastAPI에서 UploadFile 수신 시 필수 |
| **Pillow** | 이미지 처리 | 업로드된 이미지를 모델 입력 형태로 변환 (예시) |
| **torchvision** | 모델 + 전처리 | ResNet50 pretrained 모델, 이미지 transforms 포함 (예시) |

## frontend 컨테이너 (`Dockerfile.frontend`)

베이스 이미지: `python:${PYTHON_VERSION}-slim`: GPU 불필요, 경량 Python 이미지

> [hub.docker.com/_/python/tags](https://hub.docker.com/_/python/tags)

| 패키지 | 역할 | 선택 이유 |
|--------|------|-----------|
| **Streamlit** | 데모 UI | 코드 몇 줄로 인터랙티브 웹 UI 구현, ML 프로토타이핑에 최적 |
| **httpx** | HTTP 클라이언트 (프론트→API) | async + 스트리밍 지원으로 LLM 서빙 확장 시 유리 (requests 대체) |
| **Pillow** | 이미지 표시 | 업로드 이미지 미리보기 (예시) |

## nginx 컨테이너

베이스 이미지: `nginx:alpine`: 경량 리버스 프록시

> [hub.docker.com/_/nginx/tags](https://hub.docker.com/_/nginx/tags)

| 역할 | 선택 이유 |
|------|-----------|
| 리버스 프록시 | API와 프론트엔드를 단일 포트로 통합, WebSocket 프록시 |

## 로컬 학습 환경 (`pixi.toml`)

conda-forge 단일 채널, PyTorch는 플랫폼별 PyPI wheel로 설치

| 환경 | PyTorch wheel |
|------|--------------|
| Linux (CUDA 12.8) | `download.pytorch.org/whl/cu128` |
| macOS Apple Silicon | `download.pytorch.org/whl/cpu` (MPS 사용) |