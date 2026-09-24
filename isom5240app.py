import os
import re
import torch
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS

# Restrict CPU threads to optimize execution speed on Streamlit Cloud containers
torch.set_num_threads(2)

# =========================================================
# Model Loaders with Caching
# =========================================================

@st.cache_resource(show_spinner=False)
def load_caption_model():
    """Loads and caches the BLIP image captioning model."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


@st.cache_resource(show_spinner=False)
def load_story_model():
    """Loads and caches the LLM story generator pipeline."""
    return pipeline(
        "text-generation", 
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
        device_map="auto"
    )

# =========================================================
# Core Processing Functions
# =========================================================

def img2text(image_input):
    """Generates an accurate description of the uploaded image using BLIP."""
    processor, model = load_caption_model()
    
    if image_input.mode != "RGB":
        image_input = image_input.convert(mode="RGB")
        
    inputs = processor(image_input, return_tensors="pt")
    
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_new_tokens=35,
            min_new_tokens=10,
            num_beams=3,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2
        )
    
    caption = processor.decode(out[0], skip_special_tokens=True).strip().capitalize()
    if not caption.endswith('.'):
        caption += '.'
    return caption


def text2story(caption_text):
    """Generates a structured, engaging children's story (50-70 words)."""
    generator = load_story_model()
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a joyful, energetic storyteller for kids aged 3-10. "
                "Write a complete, fun short story (50 to 70 words) based strictly on the user's description. "
                "Include playful sound effects (e.g., 'Wheee!', 'Boing!'), name the main character, "
                "and ALWAYS end with a complete exciting question!"
            ),
        },
        {
            "role": "user",
            "content": f"Create a short story about: '{caption_text}'"
        },
    ]
    
    prompt = generator.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    story_result = generator(
        prompt, 
        max_new_tokens=90, 
        min_new_tokens=50,
        do_sample=True, 
        temperature=0.65,
        top_p=0.85,
        repetition_penalty=1.2
    )
    
    raw_output = story_result[0]["generated_text"]
    story_text = raw_output.split("<|assistant|>")[-1].strip()
    
    # Sentence boundary cleanup to ensure no truncation
    if not re.search(r'[.!?"]$', story_text):
        last_punct = max(story_text.rfind('.'), story_text.rfind('!'), story_text.rfind('?'))
        if last_punct != -1:
            story_text = story_text[:last_punct + 1] + " What an adventure! What would you do next?"
        else:
            story_text += "... And they lived happily ever after! Are you ready for the next adventure?"
            
    return story_text


def text2audio(story_text):
    """Converts the generated story text into an MP3 audio file."""
    audio_path = "generated_story.mp3"
    tts = gTTS(text=story_text, lang='en', slow=False)
    tts.save(audio_path)
    return audio_path

# =========================================================
# Custom UI Styling & Streamlit Main App
# =========================================================

def apply_custom_styles():
    """Applies kid-friendly pastel styling and custom UI elements."""
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #FFF5E6 0%, #FFFFFF 60%, #E8F5E9 100%);
        }
        h1 {
            color: #FF6B6B !important;
            font-family: 'Comic Sans MS', 'Chalkboard SE', cursive;
            text-align: center;
            font-size: 3rem !important;
            text-shadow: 2px 2px 0px #FFE600;
        }
        .story-card {
            background-color: #FFFFFF;
            border: 3px dashed #FF8E53;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0px 8px 16px rgba(255, 142, 83, 0.12);
            font-size: 1.2rem;
            line-height: 1.6;
            color: #2F3E46;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        div.stButton > button {
            background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
            color: white !important;
            font-size: 1.3rem !important;
            font-weight: bold !important;
            border-radius: 25px !important;
            border: none !important;
            padding: 12px 24px !important;
            box-shadow: 0 6px 15px rgba(255, 107, 107, 0.25) !important;
            width: 100%;
        }
        div.stButton > button:hover {
            background: linear-gradient(45deg, #4ECDC4, #5568FE) !important;
            transform: translateY(-2px);
        }
        .toy-grid {
            text-align: center;
            font-size: 2rem;
            margin-bottom: 10px;
            letter-spacing: 12px;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Magic Storybox AI", page_icon="🧸", layout="wide")
    apply_custom_styles()
    
    st.title("🧸 Magic Storybox AI 🚀")
    st.markdown('<div class="toy-grid">🎈 🐱 🚗 🦄 🎨 🧩</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #4A5568;'><b>Upload a picture, and let the magical story come to life! ✨</b></p>", unsafe_allow_html=True)
    st.write("---")

    # Layout splitting for better side-by-side user experience
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📸 1. Drop Your Picture")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="🌟 Your Magical Picture", use_container_width=True)
            generate_btn = st.button("✨ Spin the Magic Story Wheel! ✨")

    with col2:
        st.subheader("📖 2. Your Magic Tale")
        
        # Initialize session state variables to preserve generated outputs
        if "story" not in st.session_state:
            st.session_state.story = None
            st.session_state.caption = None
            st.session_state.audio_path = None

        if uploaded_file is not None and 'generate_btn' in locals() and generate_btn:
            with st.spinner("🔍 Step 1: Inspecting image content..."):
                st.session_state.caption = img2text(image)

            with st.spinner("✍️ Step 2: Spinning a magical child story..."):
                st.session_state.story = text2story(st.session_state.caption)

            with st.spinner("🎶 Step 3: Generating audio story..."):
                st.session_state.audio_path = text2audio(st.session_state.story)
                
            st.balloons()

        # Render outputs if available in session state
        if st.session_state.story is not None:
            st.success(f"🎨 **I see:** {st.session_state.caption}")
            st.markdown(f'<div class="story-card">{st.session_state.story}</div>', unsafe_allow_html=True)
            st.write("")
            st.subheader("🎧 Listen & Play Along:")
            st.audio(st.session_state.audio_path, format="audio/mp3")


if __name__ == "__main__":
    main()

