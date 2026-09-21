import os
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
from gtts import gTTS

# =========================================================
# Model Caching Functions
# =========================================================

@st.cache_resource
def load_blip_captioner():
    """
    Loads BLIP model and processor directly to ensure compatibility
    across Python 3.10-3.14 and transformers library versions.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_text2story_pipeline():
    """Loads and caches the GPT-2 story generation pipeline."""
    return pipeline("text-generation", model="gpt2")


# =========================================================
# Core Processing Functions
# =========================================================

def img2text(image_input):
    """
    Generates a descriptive caption from an uploaded image.
    
    Parameters:
        image_input (PIL.Image): Image provided by the user.
        
    Returns:
        str: Descriptive text caption of the image.
    """
    processor, model = load_blip_captioner()
    
    # Convert image format if needed
    if image_input.mode != "RGB":
        image_input = image_input.convert(mode="RGB")
        
    inputs = processor(image_input, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def text2story(text):
    """
    Generates a child-friendly story based on the image caption.
    
    Parameters:
        text (str): Image caption text.
        
    Returns:
        str: Story text tailored for kids aged 3-10.
    """
    generator = load_text2story_pipeline()
    prompt = f"Once upon a time, {text}. "
    
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
    Converts story text to an MP3 audio file using Google Text-to-Speech.
    
    Parameters:
        story_text (str): Generated story text.
        
    Returns:
        str: File path to saved MP3 file.
    """
    audio_path = "generated_story.mp3"
    tts = gTTS(text=story_text, lang='en', slow=False)
    tts.save(audio_path)
    return audio_path


# =========================================================
# Streamlit Interface
# =========================================================

def main():
    st.set_page_config(page_title="Magic Storyteller", page_icon="📖", layout="centered")
    
    st.title("🧙‍♂️ Magic Storyteller for Kids!")
    st.write("Upload a picture, and let the AI tell you a fun story!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Uploaded Image", use_container_width=True)
        
        if st.button("✨ Create Magic Story"):
            # Step 1: Image Captioning
            with st.spinner("1️⃣ Reading your image..."):
                caption = img2text(image)
                st.info(f"**Image Context:** {caption}")

            # Step 2: Story Generation
            with st.spinner("2️⃣ Writing a wonderful story..."):
                story = text2story(caption)
                st.subheader("📖 Your Story:")
                st.write(story)

            # Step 3: Text to Audio
            with st.spinner("3️⃣ Generating audio..."):
                audio_file_path = text2audio(story)
                st.subheader("🎧 Listen to the Story:")
                st.audio(audio_file_path, format="audio/mp3")

            st.balloons()


if __name__ == "__main__":
    main()
