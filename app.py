import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="Fruit Detection 🍓", layout="centered")

def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://cdn.pixabay.com/photo/2023/08/30/17/16/ai-generated-8223819_1280.jpg");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()
st.title("🍎 Fruit Detection App")
st.write("Upload a fruit image to detect fruit names using YOLOv11.")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload your fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        results = model(temp.name, save=True, project="runs", name="detect")

    for r in results:
        st.image(r.plot(), caption="Prediction", use_column_width=True)
        fruit_names = list(set([model.names[int(c)] for c in r.boxes.cls]))
        st.success("✅ Detected: " + ", ".join(fruit_names))
