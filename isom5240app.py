import os
import re
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS

# Set PyTorch thread limit to ensure smooth execution on Streamlit Cloud
torch.set_num_threads(2)

# =========================================================
# Model Loading with Streamlit Caching
# =========================================================

@st.cache_resource(show_spinner=False)
def load_caption_model():
    """Loads and caches the Salesforce BLIP image captioning model from Hugging Face."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


@st.cache_resource(show_spinner=False)
def load_story_model():
    """Loads and caches a lightweight GPT-2 text generation pipeline from Hugging Face."""
    return pipeline("text-generation", model="gpt2")

# =========================================================
# Core Task Functions
# =========================================================

def img2text(image_input):
    """
    Step 1: Converts an uploaded PIL image into a clean text caption using BLIP.
    """
    processor, model = load_caption_model()
    
    if image_input.mode != "RGB":
        image_input = image_input.convert(mode="RGB")
        
    inputs = processor(image_input, return_tensors="pt")
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=30,
            min_new_tokens=10,
            num_beams=3,
            no_repeat_ngram_size=2
        )
    
    caption = processor.decode(out[0], skip_special_tokens=True).strip().capitalize()
    if not caption.endswith('.'):
        caption += '.'
    return caption


def text2story(caption_text):
    """
    Step 2: Generates a short children's narrative (50-100 words) based on the image caption.
    """
    generator = load_story_model()
    
    prompt = (
        f"Once upon a time, {caption_text.lower().rstrip('.')} "
        f"Suddenly, a magical adventure began! "
    )
    
    result = generator(
        prompt,
        max_new_tokens=75,
        min_new_tokens=45,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=50256
    )
    
    story_text = result[0]["generated_text"].strip()
    
    # Clean up trailing incomplete sentences
    if not re.search(r'[.!?"]$', story_text):
        last_punct = max(story_text.rfind('.'), story_text.rfind('!'), story_text.rfind('?'))
        if last_punct != -1:
            story_text = story_text[:last_punct + 1] + " What an exciting journey!"
        else:
            story_text += "... And they lived happily ever after!"
            
    return story_text


def text2audio(story_text):
    """
    Step 3: Converts the generated story text into an MP3 audio file using gTTS.
    """
    audio_path = "generated_story.mp3"
    tts = gTTS(text=story_text, lang='en', slow=False)
    tts.save(audio_path)
    return audio_path

# =========================================================
# Streamlit Web User Interface
# =========================================================

def main():
    st.set_page_config(page_title="Magic Storybox AI", page_icon="🧸", layout="centered")
    
    st.title("🧸 Magic Storybox AI 🚀")
    st.write("Upload an image to generate an exciting story and audio narrator!")
    st.write("---")

    uploaded_file = st.file_uploader("📸 Choose an image file:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🌟 Uploaded Picture", use_container_width=True)
        
        if st.button("✨ Generate Story & Audio ✨"):
            
            # 1. Image Captioning
            with st.spinner("🔍 Analyzing your image..."):
                caption = img2text(image)
                st.info(f"**Image Caption:** {caption}")

            # 2. Story Generation
            with st.spinner("✍️ Writing a magical story..."):
                story = text2story(caption)
                st.subheader("📖 Generated Story:")
                st.write(story)

            # 3. Audio Generation
            with st.spinner("🎶 Converting story to speech..."):
                audio_file_path = text2audio(story)
                st.subheader("🎧 Listen to the Story:")
                st.audio(audio_file_path, format="audio/mp3")

            st.balloons()


if __name__ == "__main__":
    main()
