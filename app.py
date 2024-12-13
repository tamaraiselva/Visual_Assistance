import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def describe_scene(image):
    try:
        processor, model = load_blip_model()
        inputs = processor(image, return_tensors="pt")
        caption = model.generate(**inputs)
        return processor.decode(caption[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating caption: {e}"

def perform_ocr(image):
    try:
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text
    except Exception as e:
        return f"Error during OCR: {e}"

def text_to_speech_gtts(text):
    try:
        tts = gTTS(text)
        tts.save("output.mp3")
        if os.name == "nt":
            os.system("start output.mp3")
        elif os.name == "posix":
            os.system("open output.mp3" if "darwin" in os.uname().sysname.lower() else "xdg-open output.mp3")
        return "Text-to-Speech completed successfully!"
    except Exception as e:
        return f"Error during Text-to-Speech: {e}"

@st.cache_resource
def load_text_generator():
    return pipeline("text-generation", model="gpt2")

def generate_story(description):
    try:
        text_generator = load_text_generator()
        story_prompt = f"Based on the following scene description, write a short and creative story:\n\n{description}\n\nStory:"
        story = text_generator(story_prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        return story
    except Exception as e:
        return f"Error generating story: {e}"

def main():
    st.title("Enhanced Image Processing App")
    st.subheader("Perform Scene Description, OCR, and Story Generation")
    
    st.sidebar.header("Upload Images")
    scene_image_file = st.sidebar.file_uploader("Upload a scene image", type=["jpg", "png", "jpeg"])
    text_image_file = st.sidebar.file_uploader("Upload a text image for OCR", type=["jpg", "png", "jpeg"])
    
    if scene_image_file:
        st.subheader("Scene Understanding and Story Generation")
        scene_image = Image.open(scene_image_file)
        st.image(scene_image, caption="Uploaded Scene Image", use_column_width=True)
        
        with st.spinner("Describing the scene..."):
            scene_description = describe_scene(scene_image)
        st.write("Scene Description:", scene_description)
        
        if scene_description:
            with st.spinner("Generating a story based on the scene..."):
                story = generate_story(scene_description)
            st.subheader("Generated Story:")
            st.write(story)
    
    if text_image_file:
        st.subheader("OCR and Text-to-Speech")
        text_image = Image.open(text_image_file)
        st.image(text_image, caption="Uploaded Text Image", use_column_width=True)
        
        with st.spinner("Extracting text from image..."):
            extracted_text = perform_ocr(text_image)
        st.write("Extracted Text:")
        st.text(extracted_text)
        
        if extracted_text.strip():
            with st.spinner("Converting text to speech..."):
                tts_status = text_to_speech_gtts(extracted_text)
            st.success(tts_status)
        else:
            st.warning("No text found in the image to convert to speech.")

if __name__ == "__main__":
    main()
