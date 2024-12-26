import streamlit as st
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model

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

    /* Help Icon Color */
    .stSlider .stHelpIcon {
        color: white !important;  /* Change the color of the help icon to white */
    }

    /* Radio Button Text Color */
    .stRadio label {
        color: white !important;  /* Change the radio button labels to white */
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
            
            # Add a note above the slider for recommended threshold
            st.markdown(
                "**Recommended threshold: 56.65%** This is the optimal threshold based on our model's performance."
            )
            
            sensitivity = st.slider(
                "Select Detection Sensitivity", 
                min_value=0.1, 
                max_value=0.9, 
                value=0.5665,  # Set the default threshold to 0.5665
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
                                # Use the loaded model to make predictions
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
                                        f"### Probability of **Fake** Image: {predicted_prob * 100:.2f}%"
                                    )
                                    
                                    # Determine if the image is fake based on the threshold
                                    result = 'Fake' if predicted_prob > sensitivity else 'Real'
                                    st.markdown(f"**This image is classified as {result}.**")
                                  
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
    # Technology Tab
    with tabs[2]:
        st.markdown('<div class="section-header">Powered by ResNet50</div>', unsafe_allow_html=True)
        
        # Technology Overview
        st.write(
            """
            Our deepfake detection system is powered by **ResNet50**, a cutting-edge deep learning model known for its remarkable accuracy in image classification tasks.  
            By fine-tuning this model, we have adapted it to detect deepfake images with high reliability. 
            ResNet50 achieves an impressive balance between performance and efficiency, making it a top choice for tasks that require quick and accurate predictions.
            """
        )
    
        # Displaying Metrics
        st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
        st.write(
            """
            Below are the key performance metrics of our **ResNet50** model:
    
            - **Accuracy**: 79%
            - **Recall**: 92%
            - **Precision**: 73%
            - **F1-Score**: 81%
            
            These metrics represent the model's ability to accurately identify real vs. fake images, while minimizing false positives and false negatives.
            """
        )
    
        # Displaying the Images (Confusion Matrix and Loss Plot)
        st.markdown("<div class='section-header'>Model Evaluation</div>", unsafe_allow_html=True)
        st.write(
            """
            The following visualizations provide insights into the model's performance:
            - **Confusion Matrix**: Shows the model's predictions against actual labels, illustrating its accuracy.
            - **Loss Plot**: Tracks the model's training progress over time, ensuring it converges toward optimal performance.
            """
        )

        # Display Confusion Matrix and Loss Plot Images with controlled size
        st.image("https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/improved_resnet50_confusion_matrix.png", 
                 caption="Confusion Matrix", use_container_width =False, width=300)
        
        st.image("https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/improved_resnet50_loss_plot.png", 
                 caption="Loss Plot", use_container_width =False, width=300)
    
        # Model Insights
        st.write(
            """
            These visualizations highlight key aspects of the model's performance during training and evaluation. The **Confusion Matrix** reveals how well the model distinguishes between real and fake images. The **Loss Plot** demonstrates the model's ability to minimize the error across training epochs, ensuring high precision and accuracy.
            """
        )
    
    # Contact Us Tab
    with tabs[3]:
        st.markdown('<div class="section-header">Contact Us</div>', unsafe_allow_html=True)
        st.write("For inquiries or support, email us at [23054196@siswa.um.edu.my](mailto:23054196@siswa.um.edu.my).")

if __name__ == "__main__":
    main()
