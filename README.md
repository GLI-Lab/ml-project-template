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

깃허브에 새로운 저장소를 만들어, 원본 커밋 히스토리를 제외한 새 프로젝트로 시작

```bash
# 템플릿 가져오기
git clone https://github.com/gli-lab/ml-project-template.git <your-project-name>
cd <your-project-name>
rm -rf .git   # 원본 Git 이력 제거

# 새 Git 저장소로 초기화
git init
git add -A
git commit -m "init"

# GitHub에서 새 레포 생성 후 연결
git remote add origin https://github.com/<your-org>/<your-project-name>.git
git branch -M main
git push -u origin main
```

로컬에서 파이썬 환경을 설치

```bash
pixi install              # 학습 환경
pixi install -e serving   # 학습 + 서빙 환경
```

**디렉토리 구조**

아래 구조는 템플릿의 예시로, 모델이나 데이터셋이 늘어나면 같은 패턴으로 확장하면 됨됨

```
.
├── api/                     # FastAPI 서빙 코드
│   ├── main.py              # 엔드포인트 정의
│   ├── models/              # 모델별 추론 모듈
│   │   ├── base.py          # 공통 인터페이스 (ABC)
│   │   ├── resnet50/        
│   │   │   └── model.py
│   │   └── vit/             # 모델 추가 시 동일 패턴으로 확장
│   │       └── model.py
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

## 워크플로우 1단계: 로컬에서 모델 학습

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

새 모델을 추가하려면 다음 세 곳을 수정

1. **`train/{model_name}/`** : 학습 코드와 `config.yaml`(모델 하이퍼파라미터, 데이터셋별 설정, 가중치 저장 경로 등)을 작성
2. **`models/{model_name}/`** : 학습이 완료된 가중치와 `config.yaml`를 저장
3. **`api/models/`** : 모델별로 `models/{model_name}/`를 활용하는 추론 모듈을 추가

## 워크플로우 2단계: 서빙 (Docker)

Docker로 API + 프론트엔드 + 리버스 프록시를 한 번에 올림. `docker compose up` 한 줄로 동일한 환경을 어디서든 재현할 수 있음

- **api** : FastAPI 서빙 컨테이너. 모델 로드 및 추론 API를 제공하며, `api/`와 `models/` 디렉토리를 볼륨 마운트하여 코드·가중치 변경을 즉시 반영. GPU 사용 시 NVIDIA 드라이버를 컨테이너에 할당
  - 기본 포트 8000, `Dockerfile.api`로 변경 가능
- **frontend** : Streamlit UI 컨테이너. 사용자가 이미지를 업로드하고 결과를 확인하는 데모 화면을 제공. API 호출은 nginx를 경유
  - 기본 포트 8501, `Dockerfile.frontend`로 변경 가능
- **nginx** : Nginx 리버스 프록시 컨테이너. 단일 포트(기본 5001)로 `/api/*` 요청은 FastAPI로, 나머지는 Streamlit으로 라우팅
  - 기본 포트 5001, `.env`의 `NGINX_PORT`로 변경 가능

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

### 실행 방법 1: Docker 활용

`.env`는 `.gitignore` 대상이므로 `.env.example`을 복사한 뒤, `DEVICE`, `GPU_COUNT` 등을 환경에 맞게 수정

```bash
cp .env.example .env
sudo docker compose up --build
```

**주요 명령어**

```bash
sudo docker compose up --build -d             # 백그라운드 실행
sudo docker compose logs -f                   # 로그 확인
sudo docker compose restart                   # 재시작
sudo docker compose down                      # 컨테이너 삭제 (볼륨 유지)
sudo docker compose down -v --remove-orphans  # 완전 초기화
```

### 실행 방법 2: 로컬 서빙 (Docker 없이)

Docker를 사용하지 않을 때는 API와 프론트엔드를 각각 실행

```bash
pixi run -e serving api       # FastAPI (localhost:8000)
pixi run -e serving frontend  # Streamlit (localhost:8501)
```

### 기타

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

