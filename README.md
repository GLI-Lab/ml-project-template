# 🚀 ML Project Template

> Streamlit + FastAPI + Nginx + Docker + Pixi

## 사전 요구사항

- [Git](https://git-scm.com/): `sudo apt install git` (Ubuntu) / `brew install git` (macOS)
- [Pixi](https://pixi.sh/): `curl -fsSL https://pixi.sh/install.sh | sh`
- [Docker Engine](https://docs.docker.com/engine/install/): GPU 사용 시 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 추가 설치 (macOS는 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 필요)
  - [https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/)
  - [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 초기 세팅

### 1. Git 사용자 설정

```bash
# 전역 설정 (모든 레포에 적용)
git config --global user.name "gli-lab"
git config --global user.email "glilab509@gmail.com"

# 또는 이 프로젝트에만 적용하려면 --global 빼고
git config user.name "gli-lab"
git config user.email "glilab509@gmail.com"

# 설정 확인
git config --list
```

### 2. 레포 생성 및 푸시

```bash
mkdir ml-project-template && cd ml-project-template
git init

# 파일 구성 후
git add .
git commit -m "initial template"

# GitHub 웹에서 이미 만든 레포와 연결
git remote add origin https://github.com/gli-lab/ml-project-template.git
git branch -M main
git push -u origin main
```

### 3. GitHub에서 템플릿으로 지정

1. GitHub 레포 페이지 → **Settings**
2. General 섹션 → **"Template repository"** 체크박스 ON

### 4. 템플릿으로 새 프로젝트 시작

GitHub 레포 페이지에서 초록색 **"Use this template"** 버튼 클릭 → **"Create a new repository"** 선택

## 참고 문서

- [기술 스택과 선택 이유](docs/tech-stack.md)
- [LLM 서빙 확장](docs/llm-extension.md)

## 프로젝트 구조

```
.
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI
│   ├── model.py         # PyTorch 모델 로드/추론
│   ├── schemas.py       # Pydantic 스키마
│   └── requirements.txt
├── frontend/
│   ├── app.py           # Streamlit UI
│   └── requirements.txt
├── nginx/
│   └── nginx.conf       # 리버스 프록시 설정
├── models/
│   └── .gitkeep         # 모델 가중치 디렉토리 (git-ignored)
├── data/
│   └── .gitkeep         # 데이터셋 디렉토리 (git-ignored)
├── train/
│   ├── __init__.py
│   ├── train.py         # 학습 스크립트
│   ├── inference.py     # 추론 + 평가 스크립트
│   └── config.yaml      # 학습 하이퍼파라미터 설정
├── docs/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.frontend
├── pixi.toml
├── .env.example
├── .gitignore
└── README.md
```

## 1단계: 모델 학습 / 튜닝 (로컬, Docker 없이)

로컬 GPU 환경에서 pixi로 직접 실행함. 학습은 데이터셋 접근, GPU 메모리 활용, 빠른 실험 반복이 중요하므로 Docker 오버헤드 없이 로컬에서 직접 실행하는 게 효율적임.

### 환경 설치

```bash
pixi install              # 학습 환경만
pixi install -e serving   # 학습 + 서빙 (로컬 api/frontend 실행 시)
pixi install --all        # 두 환경 한 번에 설치
```

### GPU 확인

```bash
pixi run gpu-test   # PyTorch 버전 및 CUDA 사용 가능 여부 출력
pixi run gpu-info   # GPU 이름 및 메모리 정보 출력
```

### 로컬 서빙 실행 (Docker 없이)

```bash
pixi run -e serving api       # FastAPI 서버 (localhost:8000)
pixi run -e serving frontend  # Streamlit UI (localhost:8501)
```

### 학습 실행

`train/config.yaml`에서 하이퍼파라미터를 설정한 뒤 실행.

```bash
# 학습
pixi run python train/train.py

# 추론 + 평가
pixi run python train/inference.py
```

### 학습 완료 후

모델 가중치를 `models/` 디렉토리에 저장.

```bash
# 예시
cp output/best_model.pt models/model.pt
```

## 2단계: 서빙 / 데모 (Docker)

Docker로 API + 프론트엔드 + 리버스 프록시를 한 번에 올림. 환경 차이 없이 어디서든 동일하게 실행되고, 팀원이나 외부에 데모를 보여줄 때 `docker compose up` 한 줄로 끝남.

기본 예시로 ResNet50 ImageNet 이미지 분류가 포함되어 있음. 이미지를 업로드하면 top-5 분류 결과를 표시함. 모델은 torchvision에서 자동 다운로드되므로 별도 가중치 파일 불필요.

자신의 모델로 교체하려면 `api/model.py`를 수정.

### 아키텍처

기본 포트 5001, `.env`의 `NGINX_PORT`로 변경 가능함

```
Client → Nginx(:<NGINX_PORT>) ─┬→ FastAPI(:8000)    /api/*
                               └→ Streamlit(:8501)  /*
```

nginx가 `/api/` 프리픽스를 유지한 채 FastAPI로 전달하므로, FastAPI의 docs URL도 명시적으로 설정되어 있음 (`api/main.py`).

| 엔드포인트 | URL |
|---|---|
| Swagger UI | `http://localhost:<NGINX_PORT>/api/docs` |
| ReDoc | `http://localhost:<NGINX_PORT>/api/redoc` |
| Health check | `http://localhost:<NGINX_PORT>/api/health` |

### 실행

```bash
cp .env.example .env                # .env는 .gitignore 대상이라 직접 복사 필요
# .env 파일에서 DEVICE, GPU_COUNT 등 환경에 맞게 수정
docker compose up --build           # 서버 실행
```

### 주요 명령어

```bash
docker compose up --build              # 서버 실행 (컨테이너 재생성, 기존 볼륨 그대로 마운트
docker compose up --build -d           # 백그라운드 실행
docker compose logs -f                 # 로그 확인
docker compose restart                 # 재시작

# 종료
docker compose down                      # 컨테이너 삭제, 볼륨 유지
docker compose down -v --remove-orphans  # 완전 초기화 (볼륨·고아 컨테이너 포함 삭제)
```

### 모델 교체

서비스를 내리지 않고 모델을 교체할 수 있음.

```bash
# 방법 1: 리로드 API 호출 (서비스 중단 없음)
curl -X POST http://localhost:8000/api/reload

# 방법 2: API 컨테이너만 재시작
docker compose restart api
```

