import streamlit as st
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder
import folium
from folium.plugins import HeatMap
from datetime import datetime, timedelta
from streamlit_folium import folium_static
from PIL import Image

# Page Title and Config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Global CSS Styling with Background Image and button/radio styling
st.markdown("""
    <style>
        /* Custom transparent black background for the whole app */
        .css-1v0mbdj {  /* Tab content */
            background-color: rgba(0, 0, 0, 0.6);  /* Transparent black with 0.6 opacity */
        }

        /* Custom transparent black background for the radio button widget */
        .stRadio > div > label {
            background-color: rgba(0, 0, 0, 0.6);  /* Transparent black */
            color: white;
            padding: 10px;
            border-radius: 5px;
        }

        /* Custom transparent black background for the file uploader widget */
        .stFileUploader > div {
            background-color: rgba(0, 0, 0, 0.6);  /* Transparent black */
            border-radius: 5px;
        }

        /* Custom transparent background for other Streamlit elements like buttons */
        .stButton > button {
            background-color: rgba(0, 0, 0, 0.6);  /* Transparent black */
            color: white;
            border-radius: 5px;
        }

        /* Apply transparent background to other form elements if needed */
        .stTextInput > div > div {
            background-color: rgba(0, 0, 0, 0.6);  /* Transparent black */
            color: white;
            border-radius: 5px;
        }

        /* Make sure text inputs also have transparent black background */
        .stTextArea > div > div {
            background-color: rgba(0, 0, 0, 0.6);  /* Transparent black */
            color: white;
            border-radius: 5px;
        }

        /* Target the entire column and apply transparent black background */
        .stColumn {
            background-color: rgba(0, 0, 0, 0.6);  /* Transparent black */
            border-radius: 10px;
            padding: 10px;
            margin: 0 !important;  /* Remove any margin */
        }

        /* Remove gap between columns */
        .stColumns {
            display: flex !important;           /* Ensure columns are displayed as flex */
            gap: 0 !important;                  /* Remove any space (gap) between columns */
            margin: 0 !important;               /* Remove any margin around columns */
        }

        .stColumn > div {
            margin: 0 !important;               /* Remove internal margin in the column */
            padding: 0 !important;              /* Remove internal padding in the column */
        }
    </style>
""", unsafe_allow_html=True)


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
        background-color: rgba(0, 0, 0, 0.6);  /* Semi-transparent black background with 0.6 opacity */
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
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
    }
    .sub-title {
        font-size: 24px;
        font-weight: 400;
        color: #ffffff;
        text-align: center;
        margin-bottom: 40px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6);
    }

    /* Section Headers */
    .section-header {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6);
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
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.6);
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
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6);
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
# model = ResNet50(weights="imagenet")
model = load_model("improved_resnet50.keras")  # Load your custom .keras file


# Preprocess Image for Prediction
# def preprocess_image(image_file, target_size=(224, 224)):
#     """Preprocess the image for model prediction."""
#     try:
#         image = load_img(image_file, target_size=target_size)
#         image_array = img_to_array(image)
#         image_array = np.expand_dims(image_array, axis=0)
#         return preprocess_input(image_array)
#     except Exception as e:
#         st.error(f"Error processing image: {e}")
#         return None

def preprocess_image(uploaded_file, target_size=(224, 224)):
    """Preprocess image from file uploader for prediction."""
    try:
        image = Image.open(uploaded_file).convert("RGB")  # Open and ensure 3 channels (RGB)
        image = image.resize(target_size)  # Resize to match the model's input size
        image_array = img_to_array(image)  # Convert to numpy array
        image_array = image_array / 255.0  # Normalize pixel values to 0-1
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
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
    tabs = st.tabs(["About Us", "What is Deepfake", "Detection", "Technology", "Dashboard", "Contact Us"])
    
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

    # Inside the What is Deepfake Tab

    with tabs[1]:
        # What is Deepfake section
        st.markdown(
            """
            <div class="tab-content">
            <div class="section-header" style="color: #FFFFFF; font-size: 24px; font-weight: bold;">What is Deepfake?</div>
            <p style="font-size: 16px; color: #FFFFFF;">
            <b>Deepfake</b> refers to media—mostly videos or images—created using artificial intelligence (AI) to manipulate or generate realistic but fake content. 
            The term is a combination of "deep learning" (a form of AI) and "fake." Deepfakes are often used to create misleading or harmful content, such as fake videos of people saying things they never did.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Create three columns for the news items
        col1, col2, col3 = st.columns(3)
    
        # First News Item in Column 1
        with col1:
            st.markdown(
                """
               <div class="tab-content">
                <div class="news-item" style="margin-left: 0; margin-right: 0; text-align: center; height: 100%; width: 100%">
                    <img src="https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/news2.jpg" alt="Top 5 Cases of AI Deepfake Fraud Exposed in 2024" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; margin-bottom: 10px;" />
                    <a href="https://incode.com/blog/top-5-cases-of-ai-deepfake-fraud-from-2024-exposed/" target="_blank" style="font-size: 16px; color: #ffffff; text-decoration: none; font-weight: bold;">Top 5 Cases of AI Deepfake Fraud Exposed in 2024</a>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
        # Second News Item in Column 2
        with col2:
            st.markdown(
                """
                <div class="tab-content">
                <div class="news-item" style="margin-left: 0; margin-right: 0; text-align: center; height: 100%;width: 100%">
                    <img src="https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/news1.jpg" alt="AI and 2024 Elections: What to Expect" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; margin-bottom: 10px;" />
                    <a href="https://time.com/7131271/ai-2024-elections/" target="_blank" style="font-size: 16px; color: #ffffff; text-decoration: none; font-weight: bold;">AI and 2024 Elections: What to Expect</a>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
        # Third News Item in Column 3
        with col3:
            st.markdown(
                """
                <div class="tab-content">
                <div class="news-item" style="margin-left: 0; margin-right: 0; text-align: center; height: 100%;width: 100%">
                    <img src="https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/news3.jpg" alt="Deepfake CFO Scam in Hong Kong - A New Era of Fraud" style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; margin-bottom: 10px;" />
                    <a href="https://edition.cnn.com/2024/02/04/asia/deepfake-cfo-scam-hong-kong-intl-hnk/index.html" target="_blank" style="font-size: 16px; color: #ffffff; text-decoration: none; font-weight: bold;">Deepfake CFO Scam in Hong Kong - A New Era of Fraud</a>
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
        # Test Your Ability to Detect Deepfakes section
        st.markdown(
            """
            <div class="tab-content">
            <div class="section-header" style="color: #FFFFFF; font-size: 24px; font-weight: bold;">Test Your Ability to Detect Deepfakes!</div>
            <p style="font-size: 16px; color: #FFFFFF;">Let's see how good you are at detecting deepfake images! Below are 3 images. Please classify whether each one is a deepfake or not. Your score will be calculated at the end.</p>
            """,
            unsafe_allow_html=True
)
        
        # Sample Deepfake Images
        deepfake_images = [
            "https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/deepfake1.jpg",
            "https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/deepfake2.jpg",
            "https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/deepfake3.jpg"
        ]
        
        # User answers and scoring
        answers = []
        score = 0
        
        # Loop over the images
        for idx, image_url in enumerate(deepfake_images):
            # Create 3 columns: 1 for the image, 2 for the question, and 3 for extra content
            col1, col2 = st.columns([1, 2])  # Adjust column ratios as needed
            
            with col1:
                # Image column
                st.markdown(f"""
                    <div class="tab-content">
                        <div class="question-box" style="text-align: center;  ">
                            <img src="{image_url}" alt="Image {idx + 1}" style="width: 300px; height: auto; border-radius: 8px;"/>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Question column
                answer = st.radio(
                    f"Is this a deepfake? (Image {idx + 1})", 
                    ["Yes", "No", "I'm not sure"], 
                    key=f"question_{idx}", 
                    index=2  # Default to "I'm not sure"
                )
                answers.append(answer)

        
                # Calculate score (assuming correct answer is "Yes" for all images)
                if answer == "Yes":  # Assuming all images are deepfakes in this case
                    score += 1
        
            # with col3:
            #     # Extra column for additional content (optional)
            #     # You can add an explanation, a button, or leave it empty for spacing
            #     st.markdown(f"<div style='height: 200px;'></div>", unsafe_allow_html=True)  # Optional empty space
            
        # Show score and feedback
        if len(answers) == len(deepfake_images):
            st.markdown(f"Your score: {score}/3", unsafe_allow_html=True)
            if score == 3:
                st.success("Excellent! You correctly identified all the deepfakes.")
            elif score == 2:
                st.warning("Good job! You got 2 out of 3 correct.")
            else:
                st.error("Please try again! Use our deepfake detection system if you're uncertain. [Learn more about deepfakes here](https://www.youtube.com/watch?v=4JNBDwd40is)")
        


    
    # Detection Tab
    # Detection Tab
    # Detection Tab

    # Detection Tab
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
                                    
                                    # Show the radio button and comment box after the analysis
                                    st.markdown(
                                        """
                                        <div class="section-header">Report Deepfake</div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
    
                                    # Radio button for "Yes" or "No"
                                    agree = st.radio(
                                        "Would you like to report this image as a deepfake?", 
                                        ["Yes", "No"],  # Only "Yes" and "No" options
                                        index=0,  # Default to "Yes" (index 0)
                                        key="agree_radio"
                                    )
    
                                    # Comment box
                                    comment = st.text_area(
                                        "Leave a comment (optional):", 
                                        placeholder="Please enter your comment here.",
                                        key="comment_text"
                                    )
    
                                    # Submit button for comment
                                    if st.button("Submit Comment"):
                                        if comment:
                                            # Display and save the comment
                                            st.write(f"**Comment:** {comment}")
                                        else:
                                            st.warning("Please enter a comment before submitting.")
    
                                    # Handle the reporting logic after the radio selection
                                    if agree == "Yes":
                                        report_fake_image()  # Call your function to report fake images
                                        st.write(f"**Comment:** {comment}" if comment else "")
                                    elif agree == "No":
                                        st.write("Thank you for your feedback.")
                                        st.write(f"**Comment:** {comment}" if comment else "")
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
    

    


    # with tabs[2]:
    
    #     # Layout with two columns
    #     col1, col2 = st.columns([1, 2])  # Adjust column ratios for a better balance
        
    #     with col1:  # Left column for upload and detection controls
    #         st.markdown(
    #             """
    #             <div class="section-header">Image Upload</div>
    #             """, 
    #             unsafe_allow_html=True
    #         )
    #         uploaded_file = st.file_uploader("Upload a JPG, JPEG, or PNG image for detection.", type=["jpg", "jpeg", "png"])
            
    #         # Add a note above the slider for recommended threshold
    #         st.markdown(
    #             """
    #             <p><b>Recommended threshold: 56.65%</b> This is the optimal threshold based on our model's performance.</p>
    #             """, 
    #             unsafe_allow_html=True
    #         )
            
    #         sensitivity = st.slider(
    #             "Select Detection Sensitivity", 
    #             min_value=0.1, 
    #             max_value=0.9, 
    #             value=0.5665,  # Set the default threshold to 0.5665
    #             step=0.05, 
    #             help="Adjust the sensitivity of the deepfake detection model. Lower sensitivity may result in fewer false positives."
    #         )
            
    #         # Single detection button
    #         if st.button("Detect Deepfake"):
    #             if uploaded_file:
    #                 # Process the image and analyze
    #                 image_array = preprocess_image(uploaded_file)
    #                 if image_array is not None:
    #                     with st.spinner("Analyzing the image..."):
    #                         try:
    #                             # Use the loaded model to make predictions
    #                             prediction = model.predict(image_array)
    #                             # Extract the top predicted class and the corresponding probability
    #                             predicted_class = np.argmax(prediction[0])
    #                             predicted_prob = prediction[0][predicted_class]
                                
    #                             # Display results in the right column
    #                             with col2:
    #                                 st.image(
    #                                     uploaded_file, 
    #                                     caption="Uploaded Image", 
    #                                     use_container_width=False, 
    #                                     width=400  # Set the width to 400px for the uploaded image
    #                                 )
    #                                 st.markdown(
    #                                     f"### Probability of **Fake** Image: {predicted_prob * 100:.2f}%"
    #                                 )
                                    
    #                                 # Determine if the image is fake based on the threshold
    #                                 result = 'Fake' if predicted_prob > sensitivity else 'Real'
    #                                 st.markdown(f"**This image is classified as {result}.**")
                                  
    #                                 # Option to report fake
    #                                 agree = st.radio(
    #                                     "Would you like to report this image as a deepfake?", 
    #                                     ["Yes", "No"], 
    #                                     index=0
    #                                 )
    #                                 if agree == "Yes":
    #                                     report_fake_image()
    #                         except Exception as e:
    #                             st.error(f"Error during prediction: {e}")
    #                 else:
    #                     st.warning("Please upload a valid image.")
    #             else:
    #                 st.warning("Please upload an image to proceed.")
        
    #     # End transparent box for the entire tab content
    #     st.markdown(
    #         """
    #         </div> <!-- End of the transparent box -->
    #         """, 
    #         unsafe_allow_html=True
    #     )
    


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
                <br>
                <li>For more details and access to the complete code, visit our <a href="https://github.com/wwchiam/project_deepfakedetection" target="_blank"><b>GitHub Repository</b></a>.</li>
            </div>
            </div>
            """, 
            unsafe_allow_html=True
        )



    # Dashboard Tab
    # Dashboard Tab

    
    # Sample country data with submissions and coordinates
    country_data = {
        'Country': ['United States', 'Malaysia', 'India', 'China', 'Singapore', 'Russia', 'Australia', 'Canada', 'Mexico', 'Japan'],
        'Submissions': [130, 95, 70, 120, 85, 60, 40, 50, 55, 100],
        'Latitude': [37.0902, 4.2105, 20.5937, 35.8617, 1.3521, 55.7558, -25.2744, 56.1304, 23.6345, 36.2048],  # Corrected Malaysia and Singapore coords
        'Longitude': [-95.7129, 101.9758, 78.9629, 104.1954, 103.8198, 37.6176, 133.7751, -106.3468, -90.4606, 138.2529]  # Corrected Malaysia and Singapore coords
    }
    
    # Convert to DataFrame
    country_df = pd.DataFrame(country_data)
    
    
    # Dashboard Tab
    with tabs[4]:
        ##################### Statistic Section #####################
        st.markdown("""
        <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 10px;">
            <h5 style="color: white; text-align: center;">Statistic Today</h5>
        </div>
        """, unsafe_allow_html=True)
        
        # KPI 1
        kpi1, kpi2, kpi3 = st.columns(3)
        
        with kpi1:
            st.markdown("""
            <div style="background-color: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 10px;">
                <p style="color: white; text-align: center;">Total Visitors today</p>
                <h2 style="text-align: center; color: #1E90FF;">111</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi2:
            st.markdown("""
            <div style="background-color: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 10px;">
                <p style="color: white; text-align: center;">Total Submission</p>
                <h2 style="text-align: center; color: #1E90FF;">130</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with kpi3:
            st.markdown("""
            <div style="background-color: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 10px;">
                <p style="color: white; text-align: center;">% Deepfake Detected</p>
                <h2 style="text-align: center; color: #FF6347;">70%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        ##################### Filters #####################
        countries = ['United States', 'Germany', 'India', 'China', 'Brazil', 'Russia', 'Australia', 'Canada', 'Mexico', 'Japan']
        selected_countries = st.multiselect("Select Countries", countries, default=countries, key="country_filter")
        
        today = datetime.today()
        start_date = today - timedelta(days=30)
        end_date = today
        selected_date_range = st.date_input("Select Date Range", [start_date, end_date], key="date_filter")
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        ##################### Map and Table Layout #####################
        
        # Filter the data for the selected countries (same as table)
        filtered_country_df = country_df[country_df['Country'].isin(selected_countries)]
        
        # Create the map
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Create the heatmap using the filtered data
        heat_data = [[row['Latitude'], row['Longitude'], row['Submissions']] for index, row in filtered_country_df.iterrows()]
        HeatMap(heat_data).add_to(m)
        
        # Layout for the map and the table side-by-side
        map_col, table_col = st.columns([3, 2])
        
        with map_col:
            st.markdown("""
            <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 10px;">
                <h5 style="color: white;">Heatmap of Deepfake Submissions by Country</h5>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("The map below shows the intensity of deepfake video submissions across different countries.", unsafe_allow_html=True)
            folium_static(m)
        
        with table_col:
            st.markdown("""
            <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 10px;">
                <h5 style="color: white;">Submission Data</h5>
            </div>
            """, unsafe_allow_html=True)
            
            # Display filtered data in the table (this will be filtered by the country selected)
            filtered_data = filtered_country_df
            st.write(filtered_data)
            
            # Display message when no data is available
            if filtered_data.empty:
                st.warning(f"No data available for the selected countries in the selected date range.")
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        ##################### Daily Trend Section #####################
        
        st.markdown("""
        <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.5); padding: 10px; border-radius: 10px;">
            <h5 style="color: white;">Daily Trend</h5>
        </div>
        """, unsafe_allow_html=True)
        
        ##################### Simulate Data for Charts #####################
        
        # Simulate data for the chart (Visitors, Submissions, and Detections)
        num_days = 30  # Number of days for the dataset
        
        visitors = np.random.randint(500, 2000, num_days)
        submissions = np.random.randint(10, 100, num_days)
        detections = np.random.randint(5, submissions+1, num_days)
        
        # Create the dataframes
        chart_data1 = pd.DataFrame({
            'Visitors': visitors
        }, index=pd.date_range('2024-12-01', periods=num_days))
        
        chart_data2 = pd.DataFrame({
            'Deepfakes Detected': detections
        }, index=pd.date_range('2024-12-01', periods=num_days))
        
        # Convert selected_date_range to datetime64[ns] for comparison with DataFrame index
        selected_start_date, selected_end_date = selected_date_range
        selected_start_date = pd.to_datetime(selected_start_date)
        selected_end_date = pd.to_datetime(selected_end_date)
        
        # Filter data based on the selected date range
        filtered_chart_data1 = chart_data1[(chart_data1.index >= selected_start_date) & (chart_data1.index <= selected_end_date)]
        filtered_chart_data2 = chart_data2[(chart_data2.index >= selected_start_date) & (chart_data2.index <= selected_end_date)]
        
        ##################### Display Charts #####################
        
        chart1, chart2 = st.columns(2)
        
        with chart1:
            st.markdown("#### Number of Visitors", unsafe_allow_html=True)
            st.line_chart(filtered_chart_data1)
        
        with chart2:
            st.markdown("#### Number of Deepfakes Detected over Time", unsafe_allow_html=True)
            st.line_chart(filtered_chart_data2)
        
        st.markdown("<hr/>", unsafe_allow_html=True)
    
    # Contact Us Tab
    with tabs[5]:
        st.markdown(
            """
            <div class="tab-content">
                <div class="section-header" style="font-size: 24px; font-weight: bold;">Contact Us</div>
                <p style="font-size: 16px;">For inquiries or support, please contact us at:</p>
                <p style="font-size: 16px;">📧 23054196@siswa.um.edu.com</p>
                <p style="font-size: 16px;">If you have a dataset that you would like to submit for training purposes, please upload it below:</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
        # File Upload section
        uploaded_file = st.file_uploader("Upload your dataset (CSV, JSON, etc.)", type=["csv", "json", "xlsx", "txt"])
    
        if uploaded_file is not None:
            # Show file details
            st.write(f"File uploaded: {uploaded_file.name}")
    
            # You can read the file and process it here (e.g., preview the data)
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
                st.write("Preview of the uploaded dataset:")
                st.dataframe(df.head())  # Display the first few rows of the uploaded dataset
            elif uploaded_file.type == "application/json":
                data = json.load(uploaded_file)
                st.write("Preview of the uploaded JSON data:")
                st.json(data)  # Display the JSON structure
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(uploaded_file)
                st.write("Preview of the uploaded Excel dataset:")
                st.dataframe(df.head())  # Display the first few rows of the uploaded dataset
            else:
                st.write("Unsupported file type.")
    
            # After uploading, you can ask the user to submit the file or contact you for further instructions
            st.markdown(
                """
                <div class="section-header" style="font-size: 18px; font-weight: bold;">Next Steps:</div>
                <p style="font-size: 16px;">Once the dataset is uploaded, we will review it for training purposes. If you have any further questions, feel free to reach out via the contact information above.</p>
                """, 
                unsafe_allow_html=True
            )
            # Privacy notice checkbox
            privacy_accepted = st.checkbox(
                "I have read and agree to the [Privacy Notice](#) and the terms of data usage.",
                key="privacy_check"
            )
            
            # Proceed with submission only if the user agrees to the privacy notice
            if privacy_accepted:
                # Provide a unique key to the button
                if st.button("Submit Dataset", key="submit_dataset_button"):
                    # Handle dataset submission logic
                    st.success("Thank you for submitting your dataset!")
            else:
                st.warning("Please agree to the Privacy Notice before submitting your dataset.")

            

# Run the main function
if __name__ == "__main__":
    main()
