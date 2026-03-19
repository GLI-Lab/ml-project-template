import os

import httpx
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ML Image Classifier", layout="wide")
st.title("ML Image Classifier")

# Sidebar: server health
with st.sidebar:
    st.header("Server Status")
    try:
        response = httpx.get(f"{API_URL}/api/health", timeout=5)
        health = response.json()
        st.success(f"Status: {health['status']}")
        model_lines = []
        for model_name, loaded in health["models_loaded"].items():
            icon = "✅" if loaded else "❌"
            model_lines.append(f"{icon} {model_name}")
        st.success("\n\n".join(model_lines))
    except Exception as e:
        st.error(f"Server unreachable: {e}")

# Model selector
try:
    models_response = httpx.get(f"{API_URL}/api/models", timeout=5)
    available_models = models_response.json()["models"]
except Exception:
    available_models = ["resnet50"]

selected_model = st.selectbox("Model", available_models)

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
