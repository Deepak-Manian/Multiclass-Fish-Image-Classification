import streamlit as st
from PIL import Image
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Page Config & Custom Styling ---
st.set_page_config(
    page_title="Fish Species Detector",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🐟"
)

st.markdown("""
    <style>
    .main {background-color: #fafdff;}
    .reportview-container .main .block-container{padding-top:2rem;}
    .stButton>button {background-color:#27b3b3; color:white; font-weight:bold;}
    .stFileUploader>div>div {background-color:#eafcff;}
    .stImage {border-radius:7px;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: App About ---
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
**Fish Detector App**

- Upload a JPG/PNG image of a fish.
- The model will classify the species.
- Powered by MobileNet (99.4% test accuracy).
""")
st.sidebar.markdown("---")

# --- MAIN UI ---
st.title("🐟 Fish Species Classifier")
st.caption("Upload a fish image to identify the species instantly.")

uploaded = st.file_uploader(
    "Drag & drop OR click to browse. Supported: JPG, PNG",
    type=["jpg", "jpeg", "png"],
    help="Please crop to a single fish for best results."
)

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_container_width="auto", width=320)
    with st.expander("Show Advanced Model Info"):
        st.write("Model: MobileNet (fine-tuned)")
        st.write("Input size: 224x224")
        st.write("Classes: 11 fish species")
    
    # Load model once (with caching)
    @st.cache_resource
    def get_model():
        return load_model("MobileNet/mobilenet_finetuned_model.h5")
    model = get_model()

    # Load labels
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
        if isinstance(class_labels, dict):
            class_labels = [class_labels[str(i)] for i in range(len(class_labels))]
    
    # Preprocess
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    # Predict
    probs = model.predict(arr)[0]
    top3 = np.argsort(probs)[-3:][::-1]
    st.success(f"### 🎯 Prediction: {class_labels[top3[0]].replace('_',' ').title()}")
    st.progress(float(probs[top3[0]]))
    st.subheader("Top 3 Probabilities")
    
    for i in top3:
        bar = st.progress(float(probs[i]))
        st.markdown(f"**{class_labels[i].replace('_', ' ').title()}**: {probs[i]:.2%}")
    
    with st.expander("Show Raw Probabilities"):
        st.json({class_labels[i]: f"{probs[i]*100:.1f}%" for i in range(len(class_labels))})
else:
    st.info("⬆️ Start by uploading a fish image!")

# --- FOOTER ---
st.markdown(
    "<hr style='border-top:1px solid #bbb;'>"
    "<div style='text-align:center; color:gray; font-size:0.95em;'>"
    "Made by Deepak M | MobileNet Model | Fish Classification Project | 2025"
    "</div>", unsafe_allow_html=True
)
