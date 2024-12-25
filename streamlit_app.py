import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

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
    
    /* Title Section */
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
    
    /* Section Headers */
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
    }
    
    /* Adjusted Tab Styling */
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
    
    /* File Uploader Styling */
    .stFileUploader label {
        font-size: 18px;
        color: #ffffff;
        font-weight: bold;
    }
    
    /* Result Styling */
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

# Mock Prediction Logic
def mock_predict(image_array):
    """Simulate a deepfake detection model's output."""
    return np.array([[0.8]])  # Pretend it predicts a fake image with 80% probability

# Preprocess Image for Prediction
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
    st.success("Thank you for reporting. Your input will help improve our system.")

# Main Functionality
def main():
    # Tab Layout
    tabs = st.tabs(["About", "Detection", "Technology", "Contact Us"])
    
    # About Tab
    with tabs[0]:
        st.markdown('<div class="section-header">About the Platform</div>', unsafe_allow_html=True)
        st.write(
            """
            Welcome to our **Deepfake Detection System**, your trusted tool for identifying manipulated media.  
            In an age where manipulated media is becoming alarmingly common, our Deepfake Detection platform empowers users to verify the authenticity of images with just a simple upload. 
            This tool is designed to safeguard public trust, prevent misinformation, and protect against the malicious use of deepfake technology on social media.
            """
        )
        
        st.markdown(
            """
            ### Why It Matters:
            - Over 8 million deepfake attempts are shared weekly on social media.  
            - Deepfakes fuel misinformation, invade privacy, and undermine trust.  
            """
        )

        st.markdown(
            """
            ### How We Help?:
            - **Detect & Verify**: Quickly identify manipulated media using cutting-edge deep learning techniques.  
            - **Report Deepfakes**: Contribute to combating misinformation by reporting suspicious content directly through the platform.  
            - **Stay Informed**: Access resources and guides to understand and navigate the challenges of deepfake technology.  
            """
        )
    
    # Detection Tab
    with tabs[1]:
        st.markdown('<div class="section-header">Upload an Image for Detection</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])  # Input and Output Layout
        
        with col1:
            uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
            sensitivity = st.slider(
                "Select Detection Sensitivity",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Adjust the sensitivity of the detection. Lower values may reduce false positives."
            )
            if st.button("Detect Deepfake"):
                if uploaded_file:
                    image_array = preprocess_image(uploaded_file)
                    if image_array is not None:
                        with st.spinner("Analyzing the image..."):
                            try:
                                prediction = mock_predict(image_array)
                                probability = round(prediction[0][0] * 100, 2)
                                
                                # Display results in the right column
                                with col2:
                                    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                                    st.markdown(
                                        f"<div class='result'>"
                                        f"This image is likely **{'FAKE' if prediction[0][0] > sensitivity else 'REAL'}**. "
                                        f"Confidence: {probability}%"
                                        f"</div>", unsafe_allow_html=True
                                    )
                                    if st.radio("Would you like to report this image?", ["Yes", "No"], index=1) == "Yes":
                                        report_fake_image()
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")
                    else:
                        st.warning("Please upload a valid image.")
                else:
                    st.warning("Please upload an image.")
    
    # Technology Tab
    with tabs[2]:
        st.markdown('<div class="section-header">Powered by Advanced AI</div>', unsafe_allow_html=True)
        st.write(
            """
            Our deepfake detection engine is built on ResNet50, a state-of-the-art convolutional neural network, fine-tuned for precision and reliability.
            """
        )
        st.markdown("""
            ### How it works?:
            - **Transfer Learning: Utilizing the power of ImageNet pre-trained ResNet50, our model is tailored for detecting deepfakes with advanced fine-tuning.**  
            - **Diverse Datasets: Trained on a comprehensive dataset sourced from multiple platforms to enhance generalization and robustness.**
            - **Performance: Optimized to ensure accurate, fast, and scalable detection to meet real-world challenges.**
            )
    
    # Contact Us Tab
    with tabs[3]:
        st.markdown('<div class="section-header">Contact Us</div>', unsafe_allow_html=True)
        st.write("For inquiries or support, email us at [23054196@siswa.um.edu.my](mailto:23054196@siswa.um.edu.my).")

if __name__ == "__main__":
    main()
