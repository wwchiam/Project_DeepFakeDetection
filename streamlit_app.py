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
import streamlit as st

# Inject custom CSS for transparent black background on various widgets
st.markdown("""
    <style>
        /* Custom transparent black background for the whole app */
        .css-1v0mbdj {  /* Tab content */
            background-color: rgba(0, 0, 0, 0.5);  /* Transparent black */
        }

        /* Custom transparent black background for the radio button widget */
        .stRadio > div > label {
            background-color: rgba(0, 0, 0, 0.5);  /* Transparent black */
            color: white;
            padding: 10px;
            border-radius: 5px;
        }

        /* Custom transparent black background for the file uploader widget */
        .stFileUploader > div {
            background-color: rgba(0, 0, 0, 0.5);  /* Transparent black */
            border-radius: 5px;
        }

        /* Custom transparent background for other Streamlit elements like buttons */
        .stButton > button {
            background-color: rgba(0, 0, 0, 0.5);  /* Transparent black */
            color: white;
            border-radius: 5px;
        }

        /* Apply transparent background to other form elements if needed */
        .stTextInput > div > div {
            background-color: rgba(0, 0, 0, 0.5);  /* Transparent black */
            color: white;
            border-radius: 5px;
        }

        /* Make sure text inputs also have transparent black background */
        .stTextArea > div > div {
            background-color: rgba(0, 0, 0, 0.5);  /* Transparent black */
            color: white;
            border-radius: 5px;
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


    # Inside the What is Deepfake Tab

    with tabs[1]:
        # What is Deepfake section
        st.markdown(
            """
            <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 8px; margin-bottom: 0;">
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
                <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.7); margin-top: 0; margin-bottom: 0; margin-left: 0; margin-right: 0; width: 100%; height: 100%;">
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
                <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.7); margin-top: 0; margin-bottom: 0; margin-left: 0; margin-right: 0; width: 100%; height: 100%;">
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
                <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.7); margin-top: 0; margin-bottom: 0; margin-left: 0; margin-right: 0; width: 100%; height: 100%;">
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
            <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.7); margin-top: 10; margin-bottom: 0; margin-left: 0; margin-right: 0; width: 100%; height: 100%;">
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
            col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column ratios as needed
            
            with col1:
                # Image column
                st.markdown(f"""
                    <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.7); margin-top: 0; margin-bottom: 0; margin-left: 0; margin-right: 0; width: 100%; height: 100%;">
                        <div class="question-box" style="text-align: center;">
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
        
            with col3:
                # Extra column for additional content (optional)
                # You can add an explanation, a button, or leave it empty for spacing
                st.markdown(f"<div style='height: 200px;'></div>", unsafe_allow_html=True)  # Optional empty space
            
        # Show score and feedback
        if len(answers) == len(deepfake_images):
            st.markdown(f"Your score: {score}/3", unsafe_allow_html=True)
            if score == 3:
                st.success("Excellent! You correctly identified all the deepfakes.")
            elif score == 2:
                st.warning("Good job! You got 2 out of 3 correct.")
            else:
                st.error("Try again! You can improve your ability to spot deepfakes.")
        



        # st.markdown(
        #     """
        #     <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 8px;">
        #     <div class="section-header" style="color: #FFFFFF; font-size: 24px; font-weight: bold;">Test Your Ability to Detect Deepfakes!</div>
        #     <p style="font-size: 16px; color: #FFFFFF;">Let's see how good you are at detecting deepfake images! Below are 3 images. Please classify whether each one is a deepfake. Your score will be calculated at the end.</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )
    
        # # Sample Deepfake Images
        # deepfake_images = [
        #     "https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/deepfake1.jpg",
        #     "https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/deepfake2.jpg",
        #     "https://raw.githubusercontent.com/wwchiam/project_deepfakedetection/main/deepfake3.jpg"
        # ]
    
        # # User answers and scoring
        # answers = []
        # score = 0
    
        # for idx, image_url in enumerate(deepfake_images):
        #     st.markdown(f"""
        #     <div class="tab-content" style="background-color: rgba(0, 0, 0, 0.7); padding: 20px; border-radius: 8px;">
        #         <div class="question-box" style="text-align: left;">
        #             <img src="{image_url}" alt="Image {idx + 1}" style="width: 300px; height: auto; border-radius: 8px;"/>
        #         </div>
        #     </div>
        #     """, unsafe_allow_html=True)
            
        #     answer = st.radio(
        #         f"Is this a deepfake? (Image {idx + 1})", 
        #         ["Yes", "No", "I'm not sure"], 
        #         key=f"question_{idx}", 
        #         index=2  # Default to "I'm not sure"
        #     )
        #     answers.append(answer)
    
        #     # Calculate score (assuming correct answer is "Yes" for all images)
        #     if answer == "Yes":  # Assuming all images are deepfakes in this case
        #         score += 1
    
        # # Show score and feedback
        # if len(answers) == len(deepfake_images):
        #     st.markdown(f"Your score: {score}/3", unsafe_allow_html=True)
        #     if score == 3:
        #         st.success("Excellent! You correctly identified all the deepfakes.")
        #     elif score == 2:
        #         st.warning("Good job! You got 2 out of 3 correct.")
        #     else:
        #         st.error("Try again! You can improve your ability to spot deepfakes.")
        
        # # Close the transparent background wrapper div
        # st.markdown("</div>", unsafe_allow_html=True)
    


    
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


    # # Contact Us Tab
    # with tabs[4]: 
    #     st.markdown(
    #         """
    #         <div class="tab-content">
    #             <div class="section-header">Contact Us</div>
    #             <p>For inquiries or support, please contact us at:</p>
    #             <p>📧 23054196@siswa.um.edu.com"</p>
    #         </div>
            
    #         """, 
    #         unsafe_allow_html=True
    #     )
    # Contact Us Tab
    with tabs[4]:
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
                import pandas as pd
                df = pd.read_csv(uploaded_file)
                st.write("Preview of the uploaded dataset:")
                st.dataframe(df.head())  # Display the first few rows of the uploaded dataset
            elif uploaded_file.type == "application/json":
                import json
                data = json.load(uploaded_file)
                st.write("Preview of the uploaded JSON data:")
                st.json(data)  # Display the JSON structure
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                import pandas as pd
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
            
            # Optionally, you can provide a button to confirm submission
            if st.button("Submit Dataset"):
                st.success("Thank you for submitting your dataset! We will review it shortly.")
                # Here you can handle the submission logic, like saving the file or sending it via email
    
        else:
            st.write("Please upload a dataset to proceed.")
    


# Run the main function
if __name__ == "__main__":
    main()
