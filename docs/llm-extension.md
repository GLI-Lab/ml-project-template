# LLM 서빙 확장

현재 템플릿은 일반 PyTorch 추론 API 기준이며, `api/` 컨테이너는 의도적으로 가볍게 유지함.

> [hub.docker.com/r/vllm/vllm-openai/tags](https://hub.docker.com/r/vllm/vllm-openai/tags)

LLM 서빙이 필요해지면 `api/` 컨테이너에 LLM 서빙을 포함시키지 말고, 별개 컨테이너로 확장해서 `docker-compose.yml`에 **별도 서비스로 추가**하는 것을 권장함. LLM 서빙은 KV cache, 토큰 생성, GPU 메모리 관리 등 전용 최적화가 필요하기 때문에 일반 FastAPI 앱과 역할이 다름.

## LLM 전용 서버 옵션

| 서버 | 특징 | Docker 이미지 |
|------|------|--------------|
| **vLLM** (추천) | 고처리량, OpenAI 호환 API, 가장 넓은 모델 지원 | `vllm/vllm-openai` |
| **TGI** | HuggingFace 생태계 친화적 | `ghcr.io/huggingface/text-generation-inference` |
| **TensorRT-LLM** | NVIDIA 최적화 극대화, 설정 복잡 | NGC 이미지 |

## LangGraph multi-agent 확장 시 권장 구조

```
Client
  ↓
frontend (Streamlit)
  ↓
api/ (FastAPI + LangGraph)   ← 에이전트 오케스트레이션, 비즈니스 로직
  ├─→ vector DB / tools / external APIs
  ↓
llm/ (vLLM)            ← HTTP 호출, OpenAI 호환 API
```

`api/` 컨테이너의 LangGraph 코드가 같은 Docker 네트워크 안의 `llm/` 서비스를 HTTP로 호출하는 구조임. vLLM을 다른 서버로 교체해도 `api/` 코드는 그대로 유지됨.