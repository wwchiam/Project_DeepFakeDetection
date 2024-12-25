import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import os

# Page Title Appear at browser
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="auto"
)

# Global CSS

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://raw.githubusercontent.com/wwchiam/DeepFakeDetect/main/background.jpg');
        background-size: cover;
        background-position: center;
        font-family: Arial, sans-serif;
        font-size: 20px;
        color: #ffffff;
    }
    
    .title {
        font-size: 50px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-top: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    .sub-title {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 30px;
    }
    .result, .report {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        color: #ffffff;
    }

    /* Adjusting the subheader font size */
    .stSubheader {
        font-size: 18px !important; /* You can change this value to adjust font size */
        color: #ffffff !important;
    }

    /* Center tabs */
    .stTabs div[role="tablist"] {
        justify-content: center !important;
    }

    /* Adjust tab headers to match Objective font size */
    .stTabs [role="tab"] {
        font-size: 22px !important;
        font-weight: bold !important;
        color: #ffffff !important;
    }

    /* Ensure tabs and content have white text */
    .css-1cpxqw2, .css-18e3th9, .css-1n76uvr {
        color: #ffffff !important;
    }

    /* Change file uploader text color */
    .stFileUploader label {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True
)

###################################################################################################################################

# Title Section
st.markdown('<div class="title">Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Seeing is no longer believing </div>', unsafe_allow_html=True)

# Banner Image - No custom div needed, let Streamlit handle the structure
# st.image('https://raw.githubusercontent.com/wwchiam/DeepFakeDetect/main/DeepfakeBanner.jpg')

# Model Loading
@st.cache_resource
def load_deepfake_model(model_path):
    """Load the deepfake detection model."""
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            return model, None
        except Exception as e:
            return None, f"Failed to load the model. Error: {e}"
    return None, "Model file not found. Please check the path."

# Image Preprocessing
def preprocess_image(image_file, target_size=(224, 224)):
    """Preprocess the image for model prediction."""
    try:
        image = load_img(image_file, target_size=target_size)
        image_array = img_to_array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Report Fake Image
def report_fake_image():
    """Simulate reporting a deepfake image."""
    st.success("Thank you for reporting. We will use this image for future training.")

# Fancy Detection (Bounding Box and Probability Display)
def fancy_detection(image_file, prediction, threshold=0.5):
    """Simulate bounding box display and fake probability."""
    st.image(image_file, caption="Detected Face", use_container_width=True)
    probability = round(prediction[0][0] * 100, 2)
    if prediction[0][0] > threshold:
        st.markdown(f'<div class="result">This is a **fake** image. Probability: {probability}%</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result">This is a **real** image. Probability: {100 - probability}%</div>', unsafe_allow_html=True)

# Main Functionality
def main():
    # Load the model
    model_path = 'improved_vgg16.keras' 
    model, model_error = load_deepfake_model(model_path)

    if model_error:
        st.error(model_error)
        return

    # Tab Layout
    tabs = st.tabs(["About", "Detection","Deep Neural Network", "Contact Us"])
    
    # About Tab
    with tabs[0]:
        st.subheader("Detect Deepfakes Instantly")
        st.write("In an age where manipulated media is becoming alarmingly common, our Deepfake Detection platform empowers users to verify the authenticity of images with just a simple upload. This tool is designed to safeguard public trust, prevent misinformation, and protect against the malicious use of deepfake technology on social media.")
        
        st.subheader("Why It Matters:")
        st.markdown("""
        - **Over 8 million deepfake attempts flood social media weekly, spreading manipulated content and eroding online integrity. (Taeb & Chi, 2022).**  
        - **Deepfakes fuel misinformation, pose risks to privacy, and undermine trust in digital content.**
        """)
        
        st.subheader("How We Help:")
        st.markdown("""
        - **Detect & Verify**: Quickly identify manipulated media using cutting-edge deep learning techniques.  
        - **Report Deepfakes**: Contribute to combating misinformation by reporting suspicious content directly through the platform.  
        - **Stay Informed**: Access resources and guides to understand and navigate the challenges of deepfake technology.  
        """)
        

    # Detection Tab
    with tabs[1]:
        st.subheader("Upload an Image for Detection")
        uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            image_array = preprocess_image(uploaded_file)

            # Prediction
            if st.button("Detect Deepfake"):
                if image_array is not None and model is not None:
                    with st.spinner("Analyzing the image..."):
                        try:
                            prediction = model.predict(image_array)
                            fancy_detection(uploaded_file, prediction)
                            
                            agree = st.radio("Would you like to report this image as a deepfake?", ["Yes", "No"], index=1)
                            if agree == "Yes":
                                report_fake_image()
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                else:
                    st.warning("Please upload a valid image.")

    # Technology
    with tabs[2]:
        st.subheader("Cutting-Edge AI for Reliable Detection")
        st.write("Our deepfake detection engine is built on ResNet50, a state-of-the-art convolutional neural network, fine-tuned for precision and reliability." )

        st.subheader("How it works?")
        st.markdown("""
            - **Transfer Learning: Utilizing the power of ImageNet pre-trained ResNet50, our model is tailored for detecting deepfakes with advanced fine-tuning.**  
            - **Diverse Datasets: Trained on a comprehensive dataset sourced from multiple platforms to enhance generalization and robustness.**
            - **Performance: Optimized to ensure accurate, fast, and scalable detection to meet real-world challenges.**

        st.subheader("Our Training Results")
            
            """)
    
    # Contact us Tab
    with tabs[3]:
        st.subheader("Need Help?")
        st.write("Email to 23054196@siswa.um.edu.my for more information")


if __name__ == "__main__":
    main()
