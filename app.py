import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import urllib.request

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------
# Page Config
# ------------------------
st.set_page_config(
    page_title="TraceFinder - Forensic Scanner Identification",
    layout="wide"
)

# -------------------------------
# 🌈 Background Animation State
# -------------------------------
if "last_bg_value" not in st.session_state:
    st.session_state.last_bg_value = None

if "animate_bg" not in st.session_state:
    st.session_state.animate_bg = True

# ------------------------
# 🌈 Background Color Controller
# ------------------------
st.sidebar.subheader("🌈 Background Color")

bg_value = st.sidebar.slider(
    "Drag to change background",
    min_value=0,
    max_value=100,
    value=70
)

def get_bg_gradient(val):
    if val < 10:
        base = "#FFF9C2"
    elif val < 20:
        base = "#81D7FF"
    elif val < 30:
        base = "#9BF79E"
    elif val < 40:
        base = "#FF80AB"
    elif val < 50:
        base = "#9A3758"
    elif val < 60:
        base = "#D15145"
    elif val < 70:
        base = "#DE628B"
    elif val < 80:
        base = "#388E93"
    elif val < 90:
        base = "#ABABAB"
    else:
        base = "#080808"

    gradient = f"linear-gradient(135deg, {base} 0%, #000000 90%)"
    return gradient, base

selected_gradient, selected_base = get_bg_gradient(bg_value)

if st.session_state.last_bg_value != bg_value:
    st.session_state.animate_bg = True
    st.session_state.last_bg_value = bg_value
else:
    st.session_state.animate_bg = False

animation_css = "animation: shimmer 3s ease-in-out;" if st.session_state.animate_bg else ""

st.markdown(f"""
<style>
.stApp {{
    background: {selected_gradient};
    background-attachment: fixed;
    background-size: 300% 300%;
    {animation_css}
}}

@keyframes shimmer {{
    0% {{ background-position: -200% center; }}
    100% {{ background-position: 200% center; }}
}}

div[data-baseweb="slider"] > div {{
    background: linear-gradient(90deg, {selected_base}, #000000) !important;
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------
# 🎨 Progress Bar Color
# ------------------------
st.sidebar.subheader("🎨 Confidence Bar Color")

color_value = st.sidebar.slider(
    "Drag to change color",
    min_value=0,
    max_value=100,
    value=30
)

def get_color(val):
    if val < 25:
        return "#2196F3"
    elif val < 50:
        return "#4CAF50"
    elif val < 75:
        return "#FF9800"
    else:
        return "#F44336"

selected_color = get_color(color_value)

st.markdown(f"""
<style>
div[data-testid="stProgress"] > div > div > div {{
    background-color: {selected_color};
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------
# ✅ Glass-style Image Metadata Card
# ------------------------
st.markdown("""
<style>
.metadata-card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 12px;
    border: 1px solid rgba(255,255,255,0.3);
    color: #fff;
}
.metadata-title {
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 8px;
}
.metadata-row {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    padding: 3px 0;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# 🔥 Model Loading (Google Drive)
# ------------------------
# ------------------------
# 🔥 Model Loading (Google Drive)
# ------------------------
FILE_ID = "1OFWjpu7yBT5GQttM3xVXToEX_mrxPJff"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "scanner_model.h5"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading AI model… please wait (first run only)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return load_model(MODEL_PATH)

model = load_cnn_model()


class_names = ['Canon', 'Epson', 'HP', 'Xerox']

scanner_models = {
    "HP": "HP ScanJet Pro 2500",
    "Canon": "Canon DR-C240",
    "Epson": "Epson Perfection V39",
    "Xerox": "Xerox DocuMate 3125"
}

IMG_SIZE = 224

def preprocess_image(img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(img):
    return model.predict(preprocess_image(img), verbose=0)[0]

def confidence_level(conf):
    if conf >= 90:
        return "Very High"
    elif conf >= 75:
        return "High"
    elif conf >= 60:
        return "Medium"
    return "Low"

# ------------------------
# Sidebar Upload
# ------------------------
st.sidebar.title("📁 Upload Scanner Image")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# ------------------------
# Main UI
# ------------------------
st.title("🔍 TraceFinder – Forensic Scanner Identification Dashboard")
st.markdown("---")

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file:
    image = Image.open(uploaded_file)
    preds = predict_image(image)

    predictions = [(class_names[i], preds[i]*100) for i in range(len(preds))]
    predictions.sort(key=lambda x: x[1], reverse=True)
    predictions = predictions[:3]

    top_scanner, confidence = predictions[0]
    level = confidence_level(confidence)
    model_name = scanner_models[top_scanner]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

        size_kb = len(uploaded_file.getbuffer()) / 1024
        w, h = image.size

        st.markdown(f"""
        <div class="metadata-card">
            <div class="metadata-title">Image Details</div>
            <div class="metadata-row"><span>File</span><span>{uploaded_file.name}</span></div>
            <div class="metadata-row"><span>Format</span><span>{image.format}</span></div>
            <div class="metadata-row"><span>Dimensions</span><span>{w}×{h}px</span></div>
            <div class="metadata-row"><span>Mode</span><span>{image.mode}</span></div>
            <div class="metadata-row"><span>Size</span><span>{size_kb:.2f} KB</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.success(f"Brand: {top_scanner}")
        st.info(f"Model: {model_name}")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(int(confidence))
        st.caption(f"Confidence Level: {level}")

    st.info("👈 Upload a scanner image to start analysis.")
