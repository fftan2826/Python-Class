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
    Generates a clean, error-free caption using Beam Search and Repetition Penalty.
    Uses internal caching to prevent reloading the 1GB BLIP model weights on every click.
    """
    # OPTIMIZATION: Cache the BLIP loader internally
    @st.cache_resource
    def _cached_blip_loader():
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model

    processor, model = _cached_blip_loader()
    
    if image_input.mode != "RGB":
        image_input = image_input.convert(mode="RGB")
        
    inputs = processor(image_input, return_tensors="pt")
    
    # Beam search ensures full sentences and prevents vocabulary fragments
    out = model.generate(
        **inputs, 
        max_new_tokens=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        early_stopping=True
    )
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Post-processing clean-up
    caption = caption.strip().capitalize()
    if not caption.endswith('.'):
        caption += '.'
        
    return caption


def text2story(caption_text):
    """
    Generates a lively, interactive children's story (50-100 words) packed with
    sound effects, excitement, and a question for the reader.
    Uses internal caching and an automatic closure fallback to guarantee completeness.
    """
    # OPTIMIZATION: Cache the LLM pipeline loader internally
    @st.cache_resource
    def _cached_llm_loader():
        return pipeline(
            "text-generation", 
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype="auto"
        )

    generator = _cached_llm_loader()
    
    # Playful prompt designed specifically for children aged 3-10
    messages = [
        {
            "role": "system",
            "content": (
                "You are a magical, energetic children's storyteller. Write a complete, "
                "exciting short story (50 to 80 words) for kids aged 3 to 10. "
                "The story MUST directly match the provided image description. "
                "Use fun sound effects (like 'Wheee!', 'Splash!', or 'Pop!'), give the main character a name, "
                "and end with an exciting question to the child reader! Do not cut off mid-sentence."
            ),
        },
        {
            "role": "user",
            "content": f"Write a complete exciting short story based on this image description: '{caption_text}'."
        },
    ]
    
    prompt = generator.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # OPTIMIZATION: Increased max_new_tokens to 250 to give the model room to finish the story properly
    story_result = generator(
        prompt, 
        max_new_tokens=250, 
        do_sample=True, 
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    # Extract model output
    generated_output = story_result[0]["generated_text"]
    story_text = generated_output.split("<|assistant|>")[-1].strip()
    
    # OPTIMIZATION: Fallback mechanism to fix truncated sentences
    if not story_text.endswith(('.', '!', '?', '"')):
        # Find the last completed sentence
        last_punctuation = max(story_text.rfind('.'), story_text.rfind('!'), story_text.rfind('?'))
        if last_punctuation != -1:
            # Cut off the broken sentence fragment and append a classic fairy-tale ending
            story_text = story_text[:last_punctuation + 1] + " And they lived happily ever after! What do you think happens next?"
        else:
            # If no punctuation was found at all, append a clean closure
            story_text += "... And they lived happily ever after! Would you like to join their adventure?"
            
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
        /* Colorful background gradient */
        .stApp {
            background: linear-gradient(135deg, #FFEFBA 0%, #FFFFFF 50%, #E0C3FC 100%);
        }
        
        /* Main heading styling */
        h1 {
            color: #FF4B4B !important;
            font-family: 'Comic Sans MS', 'Chalkboard SE', cursive;
            text-align: center;
            font-size: 2.8rem !important;
            text-shadow: 2px 2px #FFE600;
        }
        
        /* Subheaders styling */
        h3, h2 {
            color: #6C5CE7 !important;
            font-family: 'Comic Sans MS', 'Chalkboard SE', cursive;
        }

        /* Card styling for story output */
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
        
        /* Custom styled button */
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

    # Interactive Image Upload Box
    uploaded_file = st.file_uploader("📸 Drop your favorite photo here:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display picture with rounded aesthetic
        st.image(image, caption="🌟 Your Magic Picture", use_container_width=True)
        
        # Action button
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
