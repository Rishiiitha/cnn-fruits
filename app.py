import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import base64
import io

# Load the CNN model
@st.cache_resource
def load_cnn_model():
    return load_model("fruit_meat_vegetable_cnn_model.h5")

model = load_cnn_model()
class_names = ['Fruit_479', 'Meat_500', 'Vegetable_374']

# Streamlit page configuration
st.set_page_config(page_title="CNN Image Classifier", layout="wide")

# Enhanced background animation
st.markdown("""
    <style>
    .stApp {
        background: 
            linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96c93d),
            radial-gradient(circle at top left, rgba(255, 255, 255, 0.3), transparent 10%);
        background-size: 400% 400%, 200% 200%;
        animation: gradient 3s ease infinite, pulse 5s ease-in-out infinite;
        min-height: 100vh;
        position: relative;
        overflow: hidden;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%, 0% 0%; }
        50% { background-position: 100% 50%, 100% 100%; }
        100% { background-position: 0% 50%, 0% 0%; }
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    .stApp > div {
        background: #FFD700;
        backdrop-filter: blur(5px);
        border-radius: 10px;
        padding: 2px;
    }
    h1, h2, h3, p, div {
        color: #000000;
        text-shadow: 0 0 5px rgba(20, 0,255, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Rest of your app code (webcam, file uploader, etc.) remains unchanged
st.title("Image Classifier -- ft.Groceries")
st.write("Upload an image or use your webcam to classify it into Fruit, Meat, or Vegetable.")
# ... (your existing webcam and file uploader code) ...

# Sidebar content
st.sidebar.header("About")
st.sidebar.write("""
This app uses a Convolutional Neural Network (CNN) trained to classify images into:
- Fruits
- Meat
- Vegetables
                 
CNN is better fit for image classification than RNN
""")

# Webcam capture component (only activated when chosen)
st.subheader("Capture Image from Webcam")
use_webcam = st.checkbox("Enable Webcam", key="webcam_toggle")

webcam_image = None
if use_webcam:
    webcam_html = """
    <div style="text-align: center;">
        <video id="webcam" width="640" height="480" autoplay style="border: 2px solid #ccc;"></video>
        <br>
        <button onclick="captureImage()" style="padding: 10px 20px; margin-top: 10px;">Capture Image</button>
        <canvas id="canvas" style="display: none;"></canvas>
        <script>
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');

            // Access webcam only when enabled
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    video.srcObject = stream;
                })
                .catch(err => {
                    console.error("Error accessing webcam: ", err);
                    alert("Could not access webcam. Please ensure permissions are granted.");
                });

            // Capture image from webcam
            function captureImage() {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL('image/jpeg');
                // Send image to Streamlit
                window.parent.postMessage({type: 'streamlit:setComponentValue', value: dataURL}, '*');
            }
        </script>
    </div>
    """
    webcam_image = st.components.v1.html(webcam_html, height=600)

# File uploader
uploaded_file = st.file_uploader("Or choose an image...", type=["jpg", "jpeg", "png"])

# Process image (from webcam or file upload)
img = None
if use_webcam and 'value' in st.session_state.get('_components', {}) and webcam_image:
    # Handle webcam image
    data_url = st.session_state['_components']['value']
    if data_url.startswith('data:image'):
        # Decode base64 image from webcam
        img_data = base64.b64decode(data_url.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
elif uploaded_file is not None:
    # Handle uploaded image
    img = Image.open(uploaded_file)

# Process and classify image if available
if img is not None:
    st.image(img, caption="Selected Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    predicted_class = class_names[class_index]

    st.success(f"Predicted Class: **{predicted_class}**")

    # Display prediction probabilities
    st.subheader("Prediction Probabilities")
    prob_dict = {class_names[i]: float(preds[0][i]) for i in range(len(class_names))}

    fig, ax = plt.subplots()
    ax.bar(prob_dict.keys(), prob_dict.values(), color=['green', 'red', 'orange'])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    for i, v in enumerate(prob_dict.values()):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontweight='bold')
    st.pyplot(fig)

st.markdown("---")
st.write("Built with TensorFlow CNN & Streamlit by Rishitha")