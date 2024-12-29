import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input

# Page Configuration
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Global CSS for Styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/background_v2.jpg');
        background-size: cover;
        background-position: center;
        font-family: Arial, sans-serif;
        color: #ffffff;
    }

    .title { font-size: 48px; font-weight: bold; color: #ffffff; text-align: center; margin-top: 20px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7); }
    .sub-title { font-size: 24px; font-weight: 400; color: #ffffff; text-align: center; margin-bottom: 40px; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5); }
    .section-header { font-size: 22px; font-weight: bold; color: #ffffff; margin-bottom: 10px; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5); }
    .result { font-size: 22px; font-weight: bold; text-align: center; margin-top: 20px; color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5); }
    .stSlider .st-bc { background-color: rgba(0, 0, 0, 0.7); color: white; }
    .stButton>button { background-color: red; color: white; font-size: 18px; border-radius: 8px; border: none; padding: 10px 20px; cursor: pointer; }
    .stButton>button:hover { background-color: darkred; }
    .stFileUploader label, .stRadio>label, .stTextArea>label { color: white !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Section
st.markdown('<div class="title">Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Empowering trust in digital media</div>', unsafe_allow_html=True)

# Load ResNet50 Model
model = ResNet50(weights="imagenet")

# Preprocess Image for Prediction
def preprocess_image(image_file, target_size=(224, 224)):
    try:
        image = load_img(image_file, target_size=target_size)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        return preprocess_input(image_array)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Main Functionality
def main():
    tabs = st.tabs(["About", "Usage", "Detection", "Technology", "Contact Us"])

    with tabs[0]:  # About Tab
        st.markdown(
            """
            <div class="section-header">About the Platform</div>
            <p>Welcome to our <b>Deepfake Detection System</b>, your trusted tool for identifying manipulated media. 
            This tool empowers users to verify the authenticity of images, safeguard public trust, and protect against the malicious use of deepfake technology.</p>
            <ul>
                <li><b>Detect & Verify</b>: Quickly identify manipulated media using cutting-edge deep learning techniques.</li>
                <li><b>Report Deepfakes</b>: Contribute to combating misinformation by reporting suspicious content.</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    with tabs[1]:  # Usage Tab
        st.markdown("<div class='section-header'>Usage Statistics</div>", unsafe_allow_html=True)

    with tabs[2]:  # Detection Tab
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("<div style='color:white;'>Upload an image (JPG, JPEG, PNG):</div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

            st.markdown("<div style='color:white;'>Detection Sensitivity:</div>", unsafe_allow_html=True)
            sensitivity = st.slider("", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

            detect_button = st.button("Detect Deepfake")

        with col2:
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True, width=400)

            if uploaded_file and detect_button:
                image_array = preprocess_image(uploaded_file)
                if image_array is not None:
                    with st.spinner("Analyzing the image..."):
                        prediction = model.predict(image_array)
                        predicted_class = np.argmax(prediction[0])
                        predicted_prob = prediction[0][predicted_class]
                        st.markdown(f"### Probability of **Fake** Image: {predicted_prob * 100:.2f}%")
                        result = "Fake" if predicted_prob > sensitivity else "Real"
                        st.markdown(f"**This image is classified as {result}.**")

                        report_fake = st.radio("Would you like to report this image as a deepfake?", ["Yes", "No"], index=1)
                        if report_fake == "Yes":
                            comment = st.text_area("Leave a comment (optional)", height=100)
                            if st.button("Submit Report"):
                                st.success("Thank you for reporting. Your input helps improve our system.")

    with tabs[3]:  # Technology Tab
        st.markdown(
            """
            <div class="section-header">Powered by ResNet50</div>
            <p>Our deepfake detection system is powered by <b>ResNet50</b>, a cutting-edge deep learning model 
            known for its remarkable accuracy in image classification tasks. By fine-tuning this model, 
            we have adapted it to detect deepfake images with high reliability.</p>
            <ul>
                <li>Accuracy: 79%</li>
                <li>Recall: 92%</li>
                <li>Precision: 73%</li>
                <li>F1-Score: 81%</li>
            </ul>
            <div style="text-align: center;">
                <img src="https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/improved_resnet50_confusion_matrix.png" alt="Confusion Matrix" width="400" />
                <br>
                <img src="https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/improved_resnet50_loss_plot.png" alt="Loss Plot" width="400" />
            </div>
            """,
            unsafe_allow_html=True
        )

    with tabs[4]:  # Contact Us Tab
        st.markdown(
            """
            <div class="section-header">Contact Us</div>
            <p>For inquiries or support, please reach us at:</p>
            <p>📧 23054196@siswa.um.edu.my</p>
            """,
            unsafe_allow_html=True
        )

# Run the main function
if __name__ == "__main__":
    main()
