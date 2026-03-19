# ML Project Template

> PyTorch 학습 → FastAPI 서빙 → Streamlit 데모까지 한 번에 구성하는 ML 프로젝트 템플릿
>
> **Stack:** PyTorch · FastAPI · Streamlit · Nginx · Docker · Pixi

## 사전 요구사항

- [Git](https://git-scm.com/): `sudo apt install git` (Ubuntu) / `brew install git` (macOS)
- [Pixi](https://pixi.sh/): `curl -fsSL https://pixi.sh/install.sh | sh`
- [Docker Engine](https://docs.docker.com/engine/install/): GPU 사용 시 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 추가 설치 (macOS는 [Docker Desktop](https://www.docker.com/products/docker-desktop/) 필요)
  - [https://docs.docker.com/engine/install/ubuntu/](https://docs.docker.com/engine/install/ubuntu/)
  - [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## 시작하기

원본 커밋 히스토리를 제외하고 새 프로젝트로 시작

```bash
git clone https://github.com/gli-lab/ml-project-template.git <your-project-name>
cd <your-project-name>
rm -rf .git   # 원본 Git 이력 제거

git init
git add -A
git commit -m "init"

# GitHub에서 새 레포 생성 후 연결
git remote add origin https://github.com/<your-org>/<your-project-name>.git
git branch -M main
git push -u origin main
```

환경을 설치

```bash
pixi install              # 학습 환경
pixi install -e serving   # 학습 + 서빙 환경
```

## 디렉토리 구조

```
.
├── api/                     # FastAPI 서빙 코드
│   ├── main.py              # 엔드포인트 정의
│   ├── model.py             # 모델 로드 / 추론
│   ├── schemas.py           # Pydantic 스키마
│   └── requirements.txt
├── frontend/                # Streamlit UI
│   ├── app.py
│   └── requirements.txt
├── nginx/
│   └── nginx.conf           # 리버스 프록시 설정
├── models/                  # 학습된 가중치
│   ├── resnet50/
│   │   └── model.pt
│   └── vit/                 # 모델 추가 시 동일 패턴으로 확장
│       └── model.pt
├── dataset/                 # 데이터셋
│   ├── imagenet/
│   │   ├── imagenet-simple-labels.json
│   │   └── imagenet_ko.json
│   └── coco/                # 데이터셋 추가 시 동일 패턴으로 확장
│       └── ...
├── train/
│   ├── utils.py             # 공통 유틸리티 (set_seed, transforms, load_config)
│   ├── resnet50/            # 모델별 학습 코드
│   │   ├── config.yaml      # 하이퍼파라미터
│   │   ├── train.py
│   │   └── inference.py
│   └── vit/                 # 모델 추가 시 동일 패턴으로 확장
│       ├── config.yaml
│       ├── train.py
│       └── inference.py
├── docs/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.frontend
├── pixi.toml
├── .env.example
└── .gitignore
```

## 워크플로우

### 1단계: 모델 학습 (로컬, Docker 없이)

학습은 GPU 메모리 활용과 빠른 실험 반복이 중요하므로 Pixi로 로컬에서 직접 실행

**GPU 확인**

```bash
pixi run gpu-test   # PyTorch 버전 및 CUDA 사용 가능 여부 확인
pixi run gpu-info   # GPU 이름 및 메모리 정보 확인
```

**학습 실행**

`train/{model}/config.yaml`에서 하이퍼파라미터를 설정한 뒤 실행

```bash
# -m: 프로젝트 루트를 sys.path에 추가해 패키지 간 import가 가능하게 함
pixi run python -m train.resnet50.train
pixi run python -m train.resnet50.inference
```

학습이 완료되면 가중치가 `config.yaml`의 `model_dir`에 자동 저장하도록 하면 됨 (`models/resnet50/model.pt` 등)

**새 모델 추가**

`train/{model_name}/`으로 만들고 `config.yaml`의 `model_dir`를 `models/{model_name}/`으로 변경.

---

### 2단계: 서빙 / 데모 (Docker)

Docker로 API + 프론트엔드 + 리버스 프록시를 한 번에 올린다. `docker compose up` 한 줄로 동일한 환경을 어디서든 재현할 수 있음

기본 예시는 ResNet50 ImageNet 이미지 분류다. 이미지를 업로드하면 top-5 결과를 표시하며, 가중치는 torchvision에서 자동 다운로드되므로 별도 파일이 불필요함

자신의 모델로 교체하려면 `api/model.py`와 `train/{model_name}/`을 수정하면 됨

**아키텍처**

기본 포트 5001, `.env`의 `NGINX_PORT`로 변경 가능

```
Client → Nginx(:<NGINX_PORT>) ─┬→ FastAPI(:8000)    /api/*
                               └→ Streamlit(:8501)  /*
```

nginx가 `/api/` 프리픽스를 유지한 채 FastAPI로 전달하므로 FastAPI docs URL도 명시적으로 설정되어 있음 (`api/main.py`).


| 엔드포인트        | URL                                        |
| ------------ | ------------------------------------------ |
| Swagger UI   | `http://localhost:<NGINX_PORT>/api/docs`   |
| ReDoc        | `http://localhost:<NGINX_PORT>/api/redoc`  |
| Health check | `http://localhost:<NGINX_PORT>/api/health` |


**실행**

```bash
cp .env.example .env   # .env는 .gitignore 대상이므로 직접 복사
# .env에서 DEVICE, GPU_COUNT 등 환경에 맞게 수정

sudo docker compose up --build
```

**주요 명령어**

```bash
sudo docker compose up --build -d           # 백그라운드 실행
sudo docker compose logs -f                 # 로그 확인
sudo docker compose restart                 # 재시작
sudo docker compose down                    # 컨테이너 삭제 (볼륨 유지)
sudo docker compose down -v --remove-orphans  # 완전 초기화
```

**로컬 서빙 (Docker 없이)**

```bash
pixi run -e serving api       # FastAPI (localhost:8000)
pixi run -e serving frontend  # Streamlit (localhost:8501)
```

**모델 교체 (서비스 중단 없이)**

```bash
# 방법 1: 리로드 API 호출
curl -X POST http://localhost:8000/api/reload

# 방법 2: API 컨테이너만 재시작
docker compose restart api
```

## 참고 문서

- [기술 스택과 선택 이유](docs/tech-stack.md)
- [LLM 서빙 확장](docs/llm-extension.md)
- [템플릿 레포 만들기](docs/template-setup.md)

