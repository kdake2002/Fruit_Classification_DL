import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

# Load model
model = torch.load("best.pt", map_location=torch.device('cpu'))
model.eval()

# Set background image with custom CSS
def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://cdn.pixabay.com/photo/2023/08/30/17/16/ai-generated-8223819_1280.jpg");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

st.title("🍎 Fruit Classification App")
st.write("Upload an image of a fruit and get the predicted name.")

uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing (adjust this to match your model's training pipeline)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map class index to label (update with your actual class labels)
    class_labels = ["Apple", "Banana", "Orange", "Strawberry", "Pineapple"]  # <-- Edit these
    predicted_class = class_labels[predicted.item()]
    
    st.success(f"✅ Predicted Fruit: **{predicted_class}**")
