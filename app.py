import os
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import easyocr
# from dotenv import load_dotenv
from openai import OpenAI
import time

# # Load environment variables
# load_dotenv()

# Initialize EasyOCR reader
def load_ocr_reader():
    reader = easyocr.Reader(['en'], gpu=False)
    return reader

# Function to extract text
def extract_text_from_image(image, reader):
    image_array = np.array(image)

    # If image is grayscale, convert to RGB
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
        # Convert RGBA to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

    results = reader.readtext(image_array)

    # Extract text from results and sort by vertical position
    text_blocks = []
    for (bbox, text, confidence) in results:
        if confidence > 0.3:
            y_coord = bbox[0][1]
            text_blocks.append((y_coord, text))

    # Sort by vertical position (top to bottom)
    text_blocks.sort(key=lambda x: x[0])

    extracted_text = " ".join([text for _, text in text_blocks])
    return extracted_text.strip()

def query_huggingface_llm(text):
    prompt = f"""Analyze this food label text for health insights:
    {text}
    Health Factors:
    1. Diabetes suitability:
    2. High blood pressure concerns:
    3. Heart health impact:
    4. Other health warnings:
    5. Overall health rating (1-10):
    Provide a short summary and short analysis on all health analysis factors.
    All should be one-liners. 
    Maximum 300 words"""
    
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        # api_key=os.getenv("HF_KEY"),
        api_key = st.secrets["HF_KEY"]
    )

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:fireworks-ai",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return completion.choices[0].message.content

# Streamlit app

st.set_page_config(
    page_title="NutriOptics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    
    st.image("logo-white.jpeg", use_container_width=True)

    st.markdown("""
    ---                
    # About
    NutriOptics is an AI-powered web application that provides health insights for people with dietary restrictions or medical conditions.
""")

    st.markdown("""
    ---                
    # Developed by:

    ### **Muhammad Razi**  
    <a href="https://github.com/razirizwan" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="20" style="vertical-align:middle; margin-right:5px;"></a>
    <a href="https://www.linkedin.com/in/razirizwan/" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" style="vertical-align:middle;"></a>
    """, unsafe_allow_html=True)

st.title("NutriOptics - See. Scan. Eat Healthy.")

st.info('''
    #### Instructions:
    1. Upload Image: Choose a clear photo of a food nutrition label
    2. Text Extraction: Text is extracted from the Image
    3. Analysis: The extracted text is analyzed for health insights using AI
    4. Review Results: Check the extracted text and AI health analysis
                    ''')

st.info('''
    #### Tips for best results:
    - Use clear, well-lit photos
    - Ensure the nutrition facts table is fully visible
    - Avoid blurry, angled, or shadowed images
    - Make sure text is readable and not cut off
                    ''')


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    
    with status_placeholder.container():
        st.info("**Processing your image...**")
        status_text = st.empty()
        
        status_text.write("**Status:** Uploading image...")
        time.sleep(0.5)
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        status_text.write("**Status:** Image uploaded successfully!")
        time.sleep(0.5)
        
        status_text.write("**Status:** Extracting text from image...")
        reader = load_ocr_reader()
        extracted_text = extract_text_from_image(image, reader)
        
        status_text.write("**Status:** Text extracted successfully!")
        time.sleep(0.5)
        
        status_text.write("**Status:** Analyzing health information...")
        response = query_huggingface_llm(extracted_text)
        
        status_text.write("**Status:** Analysis complete!")
        time.sleep(0.5)
    
    status_placeholder.empty()

    # Display Results
    with result_placeholder.container():
        st.success("**Analysis Complete!**")
        
        # Display the extracted text 
        with st.expander("View Extracted Text", expanded=False):
            st.write(extracted_text)
        
        # Display the health analysis
        st.subheader("Health Analysis:")
        st.write(response)