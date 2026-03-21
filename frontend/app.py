import os
from pathlib import Path

import httpx
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ML Image Classifier", layout="wide")
st.title("ML Image Classifier")

# Sidebar: settings
with st.sidebar:
    st.header("Settings")

    # Status
    server_ok = False
    models_loaded: dict[str, bool] = {}
    try:
        response = httpx.get(f"{API_URL}/api/health", timeout=5)
        health = response.json()
        server_ok = health["status"] == "ok"
        models_loaded = health.get("models_loaded", {})
    except Exception:
        pass

    if server_ok:
        st.success(f"Status: ok")
    else:
        st.error("Status: unreachable")

    # Fetch model details
    models_data: list[dict] = []
    try:
        models_response = httpx.get(f"{API_URL}/api/models", timeout=5)
        models_data = models_response.json()["models"]
    except Exception:
        pass

    available_models = [m["name"] for m in models_data] if models_data else list(models_loaded.keys()) or ["resnet50"]

    # Model selector
    st.subheader("Model")
    selected_model = st.selectbox(
        "Model",
        available_models,
        format_func=lambda m: f"{m} ({'loaded' if models_loaded.get(m) else 'not loaded'})",
        label_visibility="collapsed",
    )

    # Configuration
    selected_model_data = next((m for m in models_data if m["name"] == selected_model), None)
    if selected_model_data and selected_model_data.get("config"):
        cfg = selected_model_data["config"]
        with st.expander("Configuration", expanded=False):
            rows = "\n".join(f"| **{k}** | `{v}` |" for k, v in cfg.items())
            st.markdown(f"| Parameter | Value |\n|---|---|\n{rows}")

    # Weight source selector
    st.subheader("가중치 로드")
    weight_source = st.radio(
        "가중치 소스",
        ["Pretrained (torchvision)", "Custom (.pt/.pth)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if weight_source == "Pretrained (torchvision)":
        st.caption("서버 시작 시 로드된 기본 pretrained 가중치를 사용합니다.")

    else:  # Custom (.pt/.pth)
        browse_path = st.session_state.get("browse_path", "")
        try:
            browse_data = httpx.get(
                f"{API_URL}/api/browse", params={"path": browse_path}, timeout=5
            ).json()
        except Exception:
            browse_data = {"path": ".", "dirs": [], "weights": [], "num_weights": 0}

        st.caption(f"📂 models/{browse_data['path']}" if browse_data["path"] != "." else "📂 models/")

        if browse_path:
            if st.button("⬆ 상위 폴더", use_container_width=True, key="wt_up"):
                parent = str(Path(browse_path).parent)
                st.session_state["browse_path"] = "" if parent == "." else parent
                st.rerun()

        if browse_data["dirs"]:
            dir_sel = st.selectbox("하위 폴더", browse_data["dirs"], index=None, placeholder="폴더 선택...", key="wt_dir")
            if dir_sel:
                st.session_state["browse_path"] = f"{browse_path}/{dir_sel}".strip("/")
                st.rerun()

        if browse_data.get("weights"):
            selected_weight = st.selectbox("가중치 파일", browse_data["weights"], index=None, placeholder=".pt/.pth 파일 선택...", key="wt_file")
            if selected_weight:
                weight_file = f"models/{browse_data['path']}/{selected_weight}".replace("/./", "/")
                try:
                    resp = httpx.post(
                        f"{API_URL}/api/load-weights",
                        params={"model": selected_model, "path": weight_file},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    st.success(f"'{selected_weight}' 로드 완료")
                except httpx.HTTPStatusError as e:
                    st.error(f"Load failed: {e.response.json().get('detail', e)}")
                except Exception as e:
                    st.error(f"Load failed: {e}")
        else:
            st.caption("이 폴더에 .pt/.pth 파일이 없습니다.")

# Main: image upload and prediction
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader(f"Top-5 Predictions")
        uploaded_file.seek(0)
        try:
            response = httpx.post(
                f"{API_URL}/api/predict",
                params={"model": selected_model},
                files={"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)},
                timeout=30,
            )
            response.raise_for_status()
            predictions = response.json()["predictions"]

            for pred in predictions:
                label = pred["label"]
                label_ko = pred["label_ko"]
                confidence = pred["confidence"]
                st.write(f"**{label} ({label_ko})**")
                st.progress(confidence, text=f"{confidence * 100:.1f}%")
        except httpx.HTTPStatusError as e:
            st.error(f"Prediction failed: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
