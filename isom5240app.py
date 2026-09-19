import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Image to Story & Audio", layout="centered")

st.title("Image to Story & Audio Converter")
st.write("Upload an image to generate a caption, a story, and audio.")


# Load Hugging Face Pipeline
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
    # Implement story generation logic here (e.g., using an LLM pipeline or API)
    story_text = f"Once upon a time, there was {text}. And so the adventure began..."
    return story_text


# 3. Text to Audio
def text2audio(story_text):
    # Implement text-to-speech logic here
    audio_data = None
    return audio_data


# Streamlit UI
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Processing image and generating story..."):
        # Step 1: Captioning
        caption = img2text(image)
        st.subheader("Generated Caption:")
        st.write(caption)

        # Step 2: Story Generation
        story = text2story(caption)
        st.subheader("Generated Story:")
        st.write(story)

        # Step 3: Audio Generation
        audio = text2audio(story)
        if audio:
            st.subheader("Audio Story:")
            st.audio(audio)
