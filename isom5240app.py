import os
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS

# =========================================================
# Core Processing Functions (With Internal Caching)
# =========================================================

def img2text(image_input):
    """
    Generates a rapid, concise caption from the image using raw Greedy Search.
    OPTIMIZED: Correctly extracts the text string to guarantee zero AttributeError crashes.
    """
    @st.cache_resource
    def _cached_blip_loader():
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model

    processor, model = _cached_blip_loader()
    
    if image_input.mode != "RGB":
        image_input = image_input.convert(mode="RGB")
        
    inputs = processor(image_input, return_tensors="pt")
    
    # Ultra-fast greedy decoding configuration
    out = model.generate(
        **inputs, 
        max_new_tokens=15, 
        num_beams=1
    )
    
    # FIXED: Safely extract the first text string from the decoded list
    decoded_list = processor.batch_decode(out, skip_special_tokens=True)
    caption = decoded_list[0].strip().capitalize()
    
    if not caption.endswith('.'):
        caption += '.'
        
    return caption


def text2story(caption_text):
    """
    Generates a brief children's story using a direct, simplified pipeline.
    SPEED UP: Stripped out complex structural rendering to run seamlessly on CPU.
    """
    @st.cache_resource
    def _cached_llm_loader():
        return pipeline(
            "text-generation", 
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype="auto"
        )

    generator = _cached_llm_loader()
    
    # Simplified direct prompt for quicker token response times
    prompt = (
        f"<|system|>\nYou are a magical storyteller. Write an extremely short bedtime story "
        f"(strictly 40-50 words) for kids based on the description. Include one fun sound effect "
        f"and end with a friendly question. Keep it very brief.</s>\n"
        f"<|user|>\nWrite a story about: {caption_text}</s>\n<|assistant|>\n"
    )
    
    # Strictly capped tokens to prevent lengthy CPU calculations
    story_result = generator(
        prompt, 
        max_new_tokens=60, 
        do_sample=True, 
        temperature=0.7,
        top_p=0.85
    )
    
    # Clean split to fetch the story text instantly
    generated_text = story_result[0]["generated_text"]
    story_text = generated_text.split("<|assistant|>")[-1].strip()
    
    # Smart closure safety fallback
    if not story_text.endswith(('.', '!', '?', '"')):
        last_punctuation = max(story_text.rfind('.'), story_text.rfind('!'), story_text.rfind('?'))
        if last_punctuation != -1:
            story_text = story_text[:last_punctuation + 1] + " And they lived happily ever after!"
        else:
            story_text += "... And they lived happily ever after!"
            
    return story_text


def text2audio(story_text):
    """Converts story text into an MP3 audio file."""
    audio_path = "generated_story.mp3"
    tts = gTTS(text=story_text, lang='en', slow=False)
    tts.save(audio_path)
    return audio_path


# =========================================================
# Streamlit Interface with Kid-Friendly UI/UX
# =========================================================

def apply_custom_styles():
    """Injects colorful, child-friendly CSS styling into the Streamlit UI."""
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #FFF5E6 0%, #FFFFFF 60%, #E8F5E9 100%);
        }
        h1 {
            color: #FF6B6B !important;
            font-family: 'Comic Sans MS', 'Chalkboard SE', 'Marker Felt', cursive;
            text-align: center;
            font-size: 3.2rem !important;
            text-shadow: 3px 3px 0px #FFE600;
            margin-bottom: 5px !important;
        }
        .story-card {
            background-color: #FFFFFF;
            border: 4px dashed #FF8E53;
            border-radius: 25px;
            padding: 25px;
            box-shadow: 0px 10px 20px rgba(255, 142, 83, 0.15);
            font-size: 1.3rem;
            line-height: 1.7;
            color: #2F3E46;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        div.stButton > button {
            background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
            color: white !important;
            font-size: 1.5rem !important;
            font-weight: bold !important;
            border-radius: 35px !important;
            border: 3px solid #FFFFFF !important;
            padding: 15px 30px !important;
            box-shadow: 0 8px 20px rgba(255, 107, 107, 0.3) !important;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            width: 100%;
        }
        div.stButton > button:hover {
            transform: translateY(-4px) scale(1.02) !important;
            background: linear-gradient(45deg, #4ECDC4, #5568FE) !important;
        }
        .toy-grid {
            text-align: center;
            font-size: 2.3rem;
            margin: 15px 0px;
            letter-spacing: 15px;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Magic Storybox AI", page_icon="🧸", layout="centered")
    apply_custom_styles()
    
    st.title("🧸 Magic Storybox AI 🚀")
    st.markdown('<div class="toy-grid">🎈 🐱 🚗 🦄 🎨 🧩</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.3rem; color: #4A5568; font-family: \"Comic Sans MS\";'><b>Upload a picture, and watch the magic wizard spin an audio tale! ✨</b></p>", unsafe_allow_html=True)
    st.write("---")

    uploaded_file = st.file_uploader("📸 Drop your favorite photo here, little adventurer:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🌟 Your Magical Picture", use_container_width=True)
        
        if st.button("✨ Spin the Magic Story Wheel! ✨"):
            
            # Step 1: Image Captioning
            with st.spinner("🔍 1️⃣ Wizard is checking your picture..."):
                caption = img2text(image)
                st.success(f"🎨 **I see:** {caption}")

            # Step 2: Story Generation
            with st.spinner("✍️ 2️⃣ Shaking the magic wand to create a short tale..."):
                story = text2story(caption)
                st.subheader("📖 Story Time!")
                st.markdown(f'<div class="story-card">{story}</div>', unsafe_allow_html=True)

            # Step 3: Text to Audio
            with st.spinner("🎶 3️⃣ Humming the magical tunes..."):
                audio_file_path = text2audio(story)
                st.subheader("🎧 Listen & Play Along:")
                st.audio(audio_file_path, format="audio/mp3")

            st.balloons()


if __name__ == "__main__":
    main()

