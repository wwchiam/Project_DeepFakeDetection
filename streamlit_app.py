import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
import altair as alt
import plotly.express as px

# Page Title and Config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Global CSS Styling with Background Image
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
    .tab-content {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    .stRadio>label, .stSlider>div, .stTextArea>label, .stFileUploader>label {
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: red;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: darkred;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title Section
st.markdown('<div class="title">Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Empowering trust in digital media</div>', unsafe_allow_html=True)

# Load ResNet50 Model
model = ResNet50(weights="imagenet")

def preprocess_image(image_file, target_size=(224, 224)):
    try:
        image = load_img(image_file, target_size=target_size)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        return preprocess_input(image_array)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def report_fake_image():
    st.success("Thank you for reporting. Your input will help improve our system.")

def main():
    tabs = st.tabs(["About", "Usage", "Detection", "Technology", "Contact Us"])

    with tabs[0]:
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">About the Platform</div>
                <p>Welcome to our <b>Deepfake Detection System</b>, your trusted tool for identifying manipulated media.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tabs[1]:
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">Usage Statistics</div>
                <p>Details coming soon...</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tabs[2]:
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">Upload an Image for Detection</div>
                <p>Upload an image to check if it is a deepfake. Adjust the sensitivity threshold for more control over the detection results.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
            sensitivity = st.slider(
                "Select Detection Sensitivity",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05
            )
            detect_button = st.button("Detect Deepfake")

        with col2:
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            if uploaded_file and detect_button:
                image_array = preprocess_image(uploaded_file)
                if image_array is not None:
                    with st.spinner("Analyzing the image..."):
                        try:
                            prediction = model.predict(image_array)
                            predicted_class = np.argmax(prediction[0])
                            predicted_prob = prediction[0][predicted_class]

                            result = 'Fake' if predicted_prob > sensitivity else 'Real'
                            st.markdown(f"**This image is classified as {result}.**")

                            with st.form(key="report_form"):
                                report_fake = st.radio("Would you like to report this image as a deepfake?", ["Yes", "No"])
                                comment = st.text_area("Leave a comment (optional)")
                                submit_button = st.form_submit_button("Submit Report")

                                if submit_button:
                                    if report_fake == "Yes":
                                        report_fake_image()
                                    if comment:
                                        st.info(f"Your comment: {comment}")

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                else:
                    st.warning("Please upload a valid image.")

    with tabs[3]:
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">Technology</div>
                <p>Details about ResNet50 and performance metrics.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tabs[4]:
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">Contact Us</div>
                <p>📧 Contact: 23054196@siswa.um.edu.com</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
