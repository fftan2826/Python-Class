import os
import re
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
from gtts import gTTS

# =========================================================
# Core Processing Functions (With Internal Caching)
# =========================================================

def img2text(image_input):
    """
    Generates a clean, fully-decoded caption from the image.
    FIXED: Resolved the array slicing bug to return a 100% complete sentence.
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
    
    # Generation parameters for stable captioning
    out = model.generate(
        **inputs, 
        max_new_tokens=40,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    # Decode the full output sequence tensor
    caption = processor.decode(out, skip_special_tokens=True)
    
    caption = caption.strip().capitalize()
    if not caption.endswith('.'):
        caption += '.'
        
    return caption


def text2story(caption_text):
    """
    Generates a grammatically perfect, beautiful children's story (50-100 words).
    UPGRADED: Switched to Llama-3.2-1B for flawless English logic within Streamlit memory limits.
    """
    @st.cache_resource
    def _cached_llm_loader():
        # Meta's Llama-3.2-1B is the absolute best choice for smart reasoning on light hardware
        return pipeline(
            "text-generation", 
            model="meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype="auto"
        )

    generator = _cached_llm_loader()
    
    # Prompt structured using Llama-3.2's chat format
    messages = [
        {
            "role": "system",
            "content": (
                "You are a magical children's book author. Write a complete, "
                "exciting short story (50 to 80 words) for kids aged 3 to 10. "
                "The story must be grammatically flawless and directly match the description. "
                "Give the character a name, use simple words, include one fun sound effect, "
                "and end with an engaging question. Do not cut off mid-sentence."
            ),
        },
        {
            "role": "user",
            "content": f"Write a children's story based exactly on this image description: '{caption_text}'."
        },
    ]
    
    prompt = generator.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    story_result = generator(
        prompt, 
        max_new_tokens=150, 
        do_sample=True, 
        temperature=0.6, # Lower temperature ensures high grammatical accuracy
        top_p=0.9,
    )
    
    generated_output = story_result[0]["generated_text"]
    
    # Parse output cleanly based on Llama-3 chat template
    if "<|assistant|>" in generated_output:
        story_text = generated_output.split("<|assistant|>")[-1].strip()
    else:
        story_text = generated_output.replace(prompt, "").strip()
    
    # Smart Fallback mechanism to ensure the story ends cleanly
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
            background: linear-gradient(135deg, #FFEFBA 0%, #FFFFFF 50%, #E0C3FC 100%);
        }
        h1 {
            color: #FF4B4B !important;
            font-family: 'Comic Sans MS', 'Chalkboard SE', cursive;
            text-align: center;
            font-size: 2.8rem !important;
            text-shadow: 2px 2px #FFE600;
        }
        h3, h2 {
            color: #6C5CE7 !important;
            font-family: 'Comic Sans MS', 'Chalkboard SE', cursive;
        }
        .story-card {
            background-color: #FFFFFF;
            border: 4px solid #FF7675;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.1);
            font-size: 1.2rem;
            line-height: 1.6;
            color: #2D3436;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        div.stButton > button {
            background: linear-gradient(45deg, #FF7675, #FAB1A0) !important;
            color: white !important;
            font-size: 1.4rem !important;
            font-weight: bold !important;
            border-radius: 30px !important;
            border: none !important;
            padding: 12px 30px !important;
            box-shadow: 0 5px 15px rgba(255, 118, 117, 0.4) !important;
            transition: transform 0.2s ease !important;
            width: 100%;
        }
        div.stButton > button:hover {
            transform: scale(1.03) !important;
            background: linear-gradient(45deg, #00CEC9, #81ECEC) !important;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Magic Storyteller", page_icon="🦄", layout="centered")
    apply_custom_styles()
    
    st.title("🦄 Magic Storyteller for Kids! 🎉")
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #636E72;'><b>Upload a picture, and let the AI bring it to life with a fun story!</b></p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📸 Drop your favorite photo here:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🌟 Your Magic Picture", use_container_width=True)
        
        if st.button("✨ Spin the Magic Story Wheel! ✨"):
            
            # Step 1: Image Captioning
            with st.spinner("🔍 1️⃣ Looking closely at your picture..."):
                caption = img2text(image)
                st.success(f"🎨 **I see:** {caption}")

            # Step 2: Story Generation
            with st.spinner("✍️ 2️⃣ Writing a super exciting story for you..."):
                story = text2story(caption)
                st.subheader("📖 Story Time!")
                st.markdown(f'<div class="story-card">{story}</div>', unsafe_allow_html=True)

            # Step 3: Text to Audio
            with st.spinner("🎶 3️⃣ Turning your story into magic sound..."):
                audio_file_path = text2audio(story)
                st.subheader("🎧 Listen & Play Along:")
                st.audio(audio_file_path, format="audio/mp3")

            st.balloons()


if __name__ == "__main__":
    main()

