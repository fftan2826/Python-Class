import streamlit as st
from PIL import Image
from transformers import pipeline
import torch

st.set_page_config(page_title="AI Storyteller", layout="centered")

st.title("AI Image Storyteller")
st.write("Upload an image to generate a caption, story, and audio output.")


# Load Hugging Face Pipeline
@st.cache_resource
def load_img2text_model():
    return pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base"
    )


img2text_model = load_img2text_model()


# 1. Image to Text
def img2text(image):
    result = img2text_model(image)
    return result[0]["generated_text"]


# 2. Text to Story
def text2story(text):
    # Place your story generation logic / LLM pipeline here
    story_text = f"Once upon a time, there was {text}. And so the story began..."
    return story_text


# 3. Text to Audio
def text2audio(story_text):
    # Place your Text-to-Speech logic here
    audio_data = None
    return audio_data


# Main UI Layout
uploaded_image = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        with st.spinner("Generating caption..."):
            caption = img2text(image)

        st.subheader("1. Image Caption")
        st.write(caption)

        with st.spinner("Generating story..."):
            story = text2story(caption)

        st.subheader("2. Story")
        st.write(story)

        with st.spinner("Converting story to audio..."):
            audio = text2audio(story)

        st.subheader("3. Audio")
        if audio:
            st.audio(audio)
        else:
            st.info("Audio generation pipeline pending completion.")
