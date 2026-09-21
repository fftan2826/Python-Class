import os
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# =========================================================
# Pipeline Caching Functions
# Prevents model re-loading on every user interaction
# =========================================================

@st.cache_resource
def load_img2text_pipeline():
    """
    Loads and caches the image captioning model.
    Note: Updated to 'image-text-to-text' task as 'image-to-text' is deprecated in recent transformers.
    """
    return pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_text2story_pipeline():
    """Loads and caches the text generation model (GPT-2)."""
    return pipeline("text-generation", model="gpt2")


# =========================================================
# Core Processing Functions
# =========================================================

def img2text(image_input):
    """
    Extracts descriptive text/caption from an uploaded image.
    
    Parameters:
        image_input (PIL.Image): The image uploaded by the user.
        
    Returns:
        str: Generated caption describing the image content.
    """
    image_to_text_model = load_img2text_pipeline()
    caption_result = image_to_text_model(image_input)
    text = caption_result[0]["generated_text"]
    return text


def text2story(text):
    """
    Generates a fun, kid-friendly short story (approx. 50-100 words)
    based on the image caption.
    
    Parameters:
        text (str): The image caption/description.
        
    Returns:
        str: Expanded story narrative suitable for 3-10 year olds.
    """
    generator = load_text2story_pipeline()
    
    # Prompt structured for a cheerful children's story
    prompt = f"Once upon a time, {text}. "
    
    # Generate story with length constraints matching the 50-100 word requirement
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
    Converts story text into an MP3 audio file using Google Text-to-Speech (gTTS).
    
    Parameters:
        story_text (str): Story content to be converted into speech.
        
    Returns:
        str: File path to the generated MP3 audio file.
    """
    audio_path = "generated_story.mp3"
    tts = gTTS(text=story_text, lang='en', slow=False)
    tts.save(audio_path)
    return audio_path


# =========================================================
# Streamlit Web Application Interface
# =========================================================

def main():
    # Page configuration
    st.set_page_config(page_title="Magic Storyteller", page_icon="📖", layout="centered")
    
    # Header and kid-friendly interface description
    st.title("🧙‍♂️ Magic Storyteller for Kids!")
    st.write("Upload a picture, and let the AI tell you a fun story!")

    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image using current Streamlit API parameter
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Uploaded Image", use_container_width=True)
        
        # Action button to trigger pipeline execution
        if st.button("✨ Create Magic Story"):
            # Step 1: Image to Text
            with st.spinner("1️⃣ Reading your image..."):
                caption = img2text(image)
                st.info(f"**Image Context:** {caption}")

            # Step 2: Text to Story
            with st.spinner("2️⃣ Writing a wonderful story..."):
                story = text2story(caption)
                st.subheader("📖 Your Story:")
                st.write(story)

            # Step 3: Story to Audio
            with st.spinner("3️⃣ Generating audio..."):
                audio_file_path = text2audio(story)
                st.subheader("🎧 Listen to the Story:")
                st.audio(audio_file_path, format="audio/mp3")

            st.balloons()


if __name__ == "__main__":
    main()
