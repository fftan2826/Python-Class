import os
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ==========================================
# Caching Hugging Face Pipelines for Performance
# ==========================================

@st.cache_resource
def load_img2text_pipeline():
    """Load and cache the image-captioning model."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_text2story_pipeline():
    """Load and cache the text-generation model for generating stories."""
    return pipeline("text-generation", model="gpt2")


# ==========================================
# Core Functionalities
# ==========================================

def img2text(image_input):
    """
    Extracts descriptive text/caption from an image.
    Accepts either an image URL or a PIL Image object.
    """
    image_to_text_model = load_img2text_pipeline()
    caption_result = image_to_text_model(image_input)
    text = caption_result[0]["generated_text"]
    return text


def text2story(text):
    """
    Generates a fun, child-friendly short story (approx. 50-100 words)
    based on the provided image caption using a text generation model.
    """
    generator = load_text2story_pipeline()
    
    # Prompt structured to encourage a cheerful story suitable for 3-10 year old kids
    prompt = f"Once upon a time, {text}. "
    
    # Generate text with max_new_tokens to target roughly 50-100 words
    story_result = generator(
        prompt, 
        max_new_tokens=100, 
        min_new_tokens=50, 
        do_sample=True, 
        temperature=0.7,
        pad_token_id=50256
    )
    
    story_text = story_result[0]["generated_text"]
    return story_text


def text2audio(story_text):
    """
    Converts generated story text into an MP3 audio file using gTTS
    and returns the saved file path.
    """
    audio_path = "generated_story.mp3"
    tts = gTTS(text=story_text, lang='en', slow=False)
    tts.save(audio_path)
    return audio_path


# ==========================================
# Streamlit Main UI & Application Logic
# ==========================================

def main():
    # Page configuration
    st.set_page_config(page_title="Magic Storyteller", page_icon="📖", layout="centered")
    
    # Kid-friendly interface title and description
    st.title("🧙‍♂️ Magic Storyteller for Kids!")
    st.write("Upload a picture, and let the AI tell you a fun story!")

    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Uploaded Image", use_column_width=True)
        
        # Action button to trigger generation
        if st.button("✨ Create Magic Story"):
            with st.spinner("1️⃣ Reading your image..."):
                caption = img2text(image)
                st.info(f"**Image Context:** {caption}")

            with st.spinner("2️⃣ Writing a wonderful story..."):
                story = text2story(caption)
                st.subheader("📖 Your Story:")
                st.write(story)

            with st.spinner("3️⃣ Generating audio..."):
                audio_file_path = text2audio(story)
                st.subheader("🎧 Listen to the Story:")
                st.audio(audio_file_path, format="audio/mp3")

            st.balloons()


if __name__ == "__main__":
    main()
