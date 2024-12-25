import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import requests
from tensorflow.keras.models import load_model


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
    </style>
    """, unsafe_allow_html=True
)

# Title Section
st.markdown('<div class="title">Deepfake Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Empowering trust in digital media</div>', unsafe_allow_html=True)

# Download the model from Google Drive
url = "https://drive.google.com/uc?export=download&id=1-dE-T-0X1gEAbLR_14eXyTTvA4aHEKbY"
response = requests.get(url)

# Save the model to a local file
with open("improved_resnet50.keras", "wb") as f:
    f.write(response.content)

# Load the model
model = load_model("improved_resnet50.keras")


# Load your custom model
model = load_model('improved_resnet50.keras')

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
        st.subheader("Upload an Image for Detection")
        
        # Layout with two columns
        col1, col2 = st.columns([1, 2])  # Adjust column ratios for a better balance
        
        with col1:  # Left column for upload and detection controls
            uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
            
            sensitivity = st.slider(
                "Select Detection Sensitivity", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.5665, 
                step=0.05, 
                help="Adjust the sensitivity of the deepfake detection model. Lower sensitivity may result in fewer false positives."
            )
            
            # Single detection button
            if st.button("Detect Deepfake"):
                if uploaded_file:
                    # Process the image and analyze
                    image_array = preprocess_image(uploaded_file)
                    if image_array is not None:
                        with st.spinner("Analyzing the image..."):
                            try:
                                # Use the custom model to make predictions
                                prediction = model.predict(image_array)
                                # Extract the top predicted class and the corresponding probability
                                predicted_class = np.argmax(prediction[0])
                                predicted_prob = prediction[0][predicted_class]
                                
                                # Display results in the right column
                                with col2:
                                    st.image(
                                        uploaded_file, 
                                        caption="Uploaded Image", 
                                        use_container_width=False, 
                                        width=400  # Set the width to 400px for the uploaded image
                                    )
                                    st.markdown(
                                        f"### Prediction: **{'Fake' if predicted_prob > sensitivity else 'Real'}** image\n"
                                        f"Probability of Fake: {predicted_prob * 100:.2f}%"
                                    )
                                  
                                    # Option to report fake
                                    agree = st.radio(
                                        "Would you like to report this image as a deepfake?", 
                                        ["Yes", "No"], 
                                        index=1
                                    )
                                    if agree == "Yes":
                                        report_fake_image()
                            except Exception as e:
                                st.error(f"Error during prediction: {e}")
                    else:
                        st.warning("Please upload a valid image.")
                else:
                    st.warning("Please upload an image to proceed.")
                
    # Technology Tab
    with tabs[2]:
        st.markdown('<div class="section-header">Powered by ResNet50</div>', unsafe_allow_html=True)
        
        # Technology Overview
        st.write(
            """
            Our deepfake detection system leverages a **customized version of ResNet50**, fine-tuned specifically for detecting deepfakes. By adapting a robust and high-performance model like ResNet50, we've achieved an exceptional level of accuracy, with over **78.94% accuracy**, **91.75% recall**, and **81.31% F1 score** in real-world tests. 
            
            This cutting-edge solution allows us to quickly and reliably distinguish between real and manipulated images, empowering users to trust the content they interact with online. Our model has been rigorously tested on diverse datasets to ensure it provides reliable predictions under various conditions, setting a new benchmark in the fight against misinformation.
            """
        )
    
    # Contact Us Tab
    with tabs[3]:
        st.markdown('<div class="section-header">Contact Us</div>', unsafe_allow_html=True)
        st.write("For inquiries or support, email us at [23054196@siswa.um.edu.my](mailto:23054196@siswa.um.edu.my).")

if __name__ == "__main__":
    main()
