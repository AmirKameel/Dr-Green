import streamlit as st
from clarifai.client.model import Model
import base64

import os
from dotenv import load_dotenv

# Load environment variables from the `.env` file
load_dotenv()

# Access the PAT using os.getenv()
clarifai_pat = os.getenv("CLARIFAI_PAT")

def main():
    st.title("Plant Doctor: Save Your Plants!")

    # Text input for user query
    user_query = st.text_input("Write your question or describe the problem about your plant")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image of your plant", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Button to submit both image and query to the model
        if st.button("Diagnose Plant"):
            # Convert image to bytes
            image_bytes = uploaded_file.read()
            # Call function to process image and query
            process_image_and_query(image_bytes, user_query)

def process_image_and_query(image_bytes, query):
    # Process image and query using Clarifai model
    # Initialize Clarifai model
    model_url = "https://clarifai.com/gcp/generate/models/gemini-pro-vision"
    clarifai_model = Model(url=model_url, pat=clarifai_pat)
    
    # Construct the prompt including the instructions for the model
    prompt = "As a plant doctor, diagnose the plant issue based on the provided image and text. Provide treatment recommendations and advice for plant care."

    # Combine prompt, user query, and image bytes
    combined_data = prompt + " " + query + " " + base64.b64encode(image_bytes).decode()

    # Predict using Clarifai model
    model_prediction = clarifai_model.predict_by_bytes(combined_data.encode(), input_type="text")

    # Display model prediction
    st.write("Plant Doctor's Diagnosis:", model_prediction.outputs[0].data.text.raw)


if __name__ == "__main__":
    main()
