import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input
import altair as alt

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
    
    /* General Styling */
    .tab-content {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .title {
        font-size: 48px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-top: 20px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }
    
    .sub-title {
        font-size: 24px;
        font-weight: 400;
        color: #ffffff;
        text-align: center;
        margin-bottom: 40px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }

    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }

    .stTabs div[role="tablist"] {
        justify-content: center !important;
        gap: 20px !important;
    }
    
    .stTabs [role="tab"] {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 10px;
        padding: 10px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #4a90e2;
        color: #ffffff;
    }

    .stFileUploader label {
        font-size: 18px;
        color: #ffffff;
        font-weight: bold;
    }

    .result {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        color: #ffffff;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    </style>
    """, unsafe_allow_html=True
)

# Title Section
st.markdown('<div class="title">Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Empowering trust in digital media</div>', unsafe_allow_html=True)

# Load ResNet50 Model
model = ResNet50(weights="imagenet")

# Preprocess Image for Prediction
def preprocess_image(image_file, target_size=(224, 224)):
    """Preprocess the image for model prediction."""
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
    # Tab Layout
    tabs = st.tabs(["About", "Usage", "Detection", "Technology", "Contact Us"])
    
    # About Tab
    with tabs[0]:  
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">About the Platform</div>
                <p>Welcome to our <b>Deepfake Detection System</b>, your trusted tool for identifying manipulated media.</p>
                <p>In an age where manipulated media is becoming alarmingly common, our Deepfake Detection platform empowers users to verify the authenticity of images with just a simple upload. 
                This tool is designed to safeguard public trust, prevent misinformation, and protect against the malicious use of deepfake technology on social media.</p>
                
            <div class="section-header">Why It Matters?</div>
                <ul>
                <li>Over 8 million deepfake attempts are shared weekly on social media.</li>
                <li>Deepfakes fuel misinformation, invade privacy, and undermine trust.</li>
                </ul>
    
            <div class="section-header">How We Help</div>
                <ul>
                <li><b>Detect & Verify:</b> Quickly identify manipulated media using cutting-edge deep learning techniques.</li>
                <li><b>Report Deepfakes:</b> Contribute to combating misinformation by reporting suspicious content directly through the platform.</li>
                <li><b>Stay Informed:</b> Access resources and guides to understand and navigate the challenges of deepfake technology.</li>
                </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Usage Tab
    with tabs[1]: 
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">Usage Statistics</div>
                <p>Usage statistics and visualizations will be added here.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
         
    # Detection Tab
    with tabs[2]:
        col1, col2 = st.columns([1, 2])  # Left (upload) and right (result)

        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image (JPG, JPEG, PNG):",
                type=["jpg", "jpeg", "png"]
            )
            sensitivity = st.slider(
                "Detection Sensitivity", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.5,  
                step=0.05
            )
            detect_button = st.button("Detect Deepfake")

        with col2:
            if uploaded_file and detect_button:
                image_array = preprocess_image(uploaded_file)
                if image_array is not None:
                    with st.spinner("Analyzing..."):
                        prediction = model.predict(image_array)
                        predicted_prob = prediction[0][np.argmax(prediction)]
                        result = 'Fake' if predicted_prob > sensitivity else 'Real'
                        st.image(uploaded_file, caption="Uploaded Image")
                        st.markdown(
                            f"### Probability of Fake: {predicted_prob * 100:.2f}%",
                            unsafe_allow_html=True
                        )
                        st.markdown(f"**This image is classified as {result}.**", unsafe_allow_html=True)
                        report_fake = st.selectbox("Report this image?", ["No", "Yes"])
                        if st.button("Submit Report"):
                            st.success("Thank you for your feedback!")

    # Technology Tab
    with tabs[3]: 
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">Powered by ResNet50</div>
                <p>Our system is powered by <b>ResNet50</b>, a deep learning model for image classification.</p>
                <div class="section-header">Performance Metrics</div>
                <ul>
                    <li>Accuracy: 79%</li>
                    <li>Recall: 92%</li>
                    <li>Precision: 73%</li>
                    <li>F1-Score: 81%</li>
                </ul>
                <p><b>Confusion Matrix:</b></p>
                <img src="https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/improved_resnet50_confusion_matrix.png" width="400" />
                <p><b>Learning Curve:</b></p>
                <img src="https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/improved_resnet50_loss_plot.png" width="400" />
            </div>
            """, 
            unsafe_allow_html=True
        )

    # Contact Us Tab
    with tabs[4]: 
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">Contact Us</div>
                <p>Email: 23054196@siswa.um.edu.com</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Run the main function
if __name__ == "__main__":
    main()
