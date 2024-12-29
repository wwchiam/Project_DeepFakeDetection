import streamlit as st
import numpy as np
import pandas as pd
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

# Global CSS Styling with Background Image and button/radio styling
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
    
    /* Container for Tab Content */
    .tab-content {
        background-color: rgba(0, 0, 0, 0.6);  /* Semi-transparent black background */
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
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

    /* Styling for Radio Buttons */
    .stRadio label {
        color: white !important;  /* Always white font for radio buttons */
        font-size: 18px;
        font-weight: bold;
    }

    /* Styling for Buttons (Make all buttons red with white text) */
    .stButton button {
        background-color: red !important;  /* Red background */
        color: white !important;  /* White text */
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        width: 150px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: darkred !important;  /* Darker red on hover */
    }

    /* Styling for Slider (Ensure slider labels have white text) */
    .stSlider label {
        color: white !important;  /* White font for slider labels */
        font-size: 18px;
        font-weight: bold;
    }

    /* Ensure Radio button options (Yes/No) always have white text */
    .stRadio div div div span {
        color: white !important; /* White text for options in radio button */
    }

    /* Ensure selected radio button option text is also white */
    .stRadio div[aria-checked="true"] span {
        color: white !important; /* Selected option text is white */
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
    tabs = st.tabs(["About Us", "What is Deepfake", "Detection", "Technology", "Contact Us"])
    
    with tabs[0]:  # About Tab
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
    
            <div class="section-header">How can we help</div>
                <li><b>Detect & Verify</b>: Quickly identify manipulated media using cutting-edge deep learning techniques.</li>
                <li><b>Report Deepfakes</b>: Contribute to combating misinformation by reporting suspicious content directly through the platform.</li>
                <li><b>Stay Informed</b>: Access resources and guides to understand and navigate the challenges of deepfake technology.</li>
            </div>
            
            """, 
            unsafe_allow_html=True
        )

    # # Usage Tab
    # with tabs[1]: 
    #     st.markdown(
    #         """
    #         <div class="tab-content">
    #             <div class="section-header">Usage Statistic</div>
    #             <p>Thinking...</p>
    #         </div>
            
    #         """, 
    #         unsafe_allow_html=True
    #     )
         
    # Usage Tab
    # Usage Tab
    # Inside the What is Deepfake Tab
    with tabs[1]: 
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">What is Deepfake?</div>
                <p><b>Deepfake</b> refers to media—mostly videos or images—created using artificial intelligence (AI) to manipulate or generate realistic but fake content. 
                The term is a combination of "deep learning" (a form of AI) and "fake." Deepfakes are often used to create misleading or harmful content, such as fake videos of people saying things they never did.</p>
                
            <div class="section-header">Test Your Ability to Detect Deepfakes!</div>
                <p>Let's see how good you are at detecting deepfake images! Below are 3 images. Please classify whether each one is a deepfake or not. Your score will be calculated at the end.</p>
            """, 
            unsafe_allow_html=True
        )

            # Display News Links with Icons
        news_links = [
            {
                'title': 'AI and 2024 Elections: What to Expect', 
                'url': 'https://time.com/7131271/ai-2024-elections/',
                'icon': 'https://upload.wikimedia.org/wikipedia/commons/a/a3/Time_logo.svg'  # Time Logo
            },
            {
                'title': 'Top 5 Cases of AI Deepfake Fraud Exposed in 2024',
                'url': 'https://incode.com/blog/top-5-cases-of-ai-deepfake-fraud-from-2024-exposed/',
                'icon': 'https://upload.wikimedia.org/wikipedia/commons/e/e6/Incode_Logo_2x.png'  # Incode Logo
            },
            {
                'title': 'Deepfake CFO Scam in Hong Kong - A New Era of Fraud',
                'url': 'https://edition.cnn.com/2024/02/04/asia/deepfake-cfo-scam-hong-kong-intl-hnk/index.html',
                'icon': 'https://upload.wikimedia.org/wikipedia/commons/4/4d/CNN_logo.svg'  # CNN Logo
            }
        ]
    
        for article in news_links:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <img src="{article['icon']}" alt="{article['title']}" width="30" style="margin-right: 10px;" />
                    <a href="{article['url']}" target="_blank" style="font-size: 18px; color: #4a90e2; text-decoration: none;">
                        {article['title']}
                    </a>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        
        # Sample Deepfake Images (replace with actual images you want to use for the test)
        deepfake_images = [
            "https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/deepfake1.jpg",
            "https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/deepfake2.jpg",
            "https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/deepfake3.jpg"
        ]
        
        # Add CSS for background styling and hiding the "Dummy" option
        st.markdown(
            """
            <style>
                /* Style for the background of each question */
                .question-box {
                    background-color: #ffffff;  /* White background */
                    padding: 10px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                }
                
                /* Hide the dummy radio button */
                div[role=radiogroup] .st-cs:first-child {
                    visibility: hidden;
                    height: 0px;
                }
            </style>
            """, 
            unsafe_allow_html=True
        )
    
        # User answers
        answers = []
        score = 0
    
        for idx, image_url in enumerate(deepfake_images):
            st.image(image_url, caption=f"Image {idx + 1}", width=400)
            
            # Wrap each question in a div with the 'question-box' class for styling
            with st.markdown(
                """
                <div class="question-box">
                """, 
                unsafe_allow_html=True
            ):
                # Actual deepfake detection radio buttons (Yes, No, and Dummy)
                answer = st.radio(
                    f"Is this a deepfake? (Image {idx + 1})", 
                    ["Yes", "No", "I'm not sure"],  # Add "Dummy" as the third option
                    key=f"question_{idx}",  # Each question should have a unique key
                    index=2  # Default to "Dummy" (index 2)
                )
    
            # Store the answers
            answers.append(answer)
    
            # Calculate score (assumes the correct answer is "Yes" for all images)
            if answer == "Yes":  # Assuming all images are deepfakes in this case
                score += 1
    
        if len(answers) == len(deepfake_images):
            st.markdown(f"Your score: {score}/3")
            if score == 3:
                st.success("Excellent! You correctly identified all the deepfakes.")
            elif score == 2:
                st.warning("Good job! You got 2 out of 3 correct.")
            else:
                st.error("Try again! You can improve your ability to spot deepfakes.")
        
        # End of the tab content
        st.markdown(
            """
            </div> <!-- End of the transparent box -->
            """, 
            unsafe_allow_html=True
        )



    
    # Detection Tab
    # Detection Tab
    with tabs[2]:
    
        # Layout with two columns
        col1, col2 = st.columns([1, 2])  # Adjust column ratios for a better balance
        
        with col1:  # Left column for upload and detection controls
            st.markdown(
                """
                <div class="section-header">Image Upload</div>
                """, 
                unsafe_allow_html=True
            )
            uploaded_file = st.file_uploader("Upload a JPG, JPEG, or PNG image for detection.", type=["jpg", "jpeg", "png"])
            
            # Add a note above the slider for recommended threshold
            st.markdown(
                """
                <p><b>Recommended threshold: 56.65%</b> This is the optimal threshold based on our model's performance.</p>
                """, 
                unsafe_allow_html=True
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
        
        # End transparent box for the entire tab content
        st.markdown(
            """
            </div> <!-- End of the transparent box -->
            """, 
            unsafe_allow_html=True
        )
    


    # Technology Tab
    with tabs[3]: 
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header">Powered by ResNet50</div>
                <p>Our deepfake detection system is powered by <b>ResNet50</b>, a cutting-edge deep learning model known for its remarkable accuracy in image classification tasks.
                By fine-tuning this model, we have adapted it to detect deepfake images with high reliability. 
                ResNet50 achieves an impressive balance between performance and efficiency, making it a top choice for tasks that require quick and accurate predictions.</p>
                
            <div class="section-header">Model Performance</div>
                <p>These metrics represent the model's ability to accurately identify real vs. fake images, while minimizing false positives and false negatives.</p>
                <ul>
                    <li>Accuracy: 79% </li>
                    <li>Recall: 92% </li>
                    <li>Precision: 73% </li>
                    <li>F1-Score: 81% </li>
                </ul>
        
            <div class="section-header">Model Evaluation</div>
            <div style="text-align: left;">
                <ul>
                <li><b>Confusion Matrix</b>: Shows the model's predictions against actual labels, illustrating its accuracy.</li>
                </ul>
                <img src="https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/improved_resnet50_confusion_matrix.png" alt="Confusion Matrix" width="400" />
                <br>
            </div>
                
            <div style="text-align: left;">
                <ul>
                <br>
                <li><b>Learning Curve</b>: Tracks the model's training progress over time, ensuring it converges toward optimal performance.</li>
                </ul>
                <img src="https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/improved_resnet50_loss_plot.png" alt="Loss Plot" width="400" />
            </div>
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
                <p>For inquiries or support, please contact us at:</p>
                <p>📧 23054196@siswa.um.edu.com"</p>
            </div>
            
            """, 
            unsafe_allow_html=True
        )



# Run the main function
if __name__ == "__main__":
    main()
