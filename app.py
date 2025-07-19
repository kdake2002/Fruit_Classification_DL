import streamlit as st
from PIL import Image
import tempfile
from ultralytics import YOLO
import os

# Set Streamlit page config and background
st.set_page_config(page_title="Fruit Detector 🍎", layout="centered")

def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://cdn.pixabay.com/photo/2023/08/30/17/16/ai-generated-8223819_1280.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

st.title("🍍 Fruit Detection App (YOLOv11)")
st.write("Upload a fruit image and detect fruit names using a YOLOv11 model.")

# Load model
model = YOLO("best.pt")  # Already trained YOLOv11 model

# Upload
uploaded_file = st.file_uploader("Upload your fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        results = model(temp_file.name)

    # Visualize prediction
    for r in results:
        st.image(r.plot(), caption="Predictions", use_column_width=True)

        # Show detected fruit classes
        fruit_names = list(set([model.names[int(cls)] for cls in r.boxes.cls]))
        st.success(f"✅ Detected Fruits: {', '.join(fruit_names)}")
