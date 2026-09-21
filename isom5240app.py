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
    Loads BLIP model and processor for extracting accurate captions from images.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_text2story_pipeline():
    """
    Loads TinyLlama, a lightweight instruct model that adheres closely to prompt instructions.
    """
    return pipeline(
        "text-generation", 
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto"
    )


# =========================================================
# Core Processing Functions
# =========================================================

def img2text(image_input):
    """
    Generates a descriptive caption from an uploaded image.
    """
    processor, model = load_blip_captioner()
    
    # Ensure image is in RGB format
    if image_input.mode != "RGB":
        image_input = image_input.convert(mode="RGB")
        
    inputs = processor(image_input, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def text2story(caption_text):
    """
    Generates an accurate, child-friendly story based strictly on the image caption.
    """
    generator = load_text2story_pipeline()
    
    # Formulate a structured prompt using TinyLlama's chat template format
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly children's storyteller. Write a short, fun, and warm "
                "story for young children (3-10 years old). The story MUST directly describe and "
                "be based on what is happening in the provided image caption. Keep it between 50 and 80 words."
            ),
        },
        {
            "role": "user",
            "content": f"Write a children's story based on this image description: '{caption_text}'."
        },
    ]
    
    prompt = generator.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    story_result = generator(
        prompt, 
        max_new_tokens=120, 
        do_sample=True, 
        temperature=0.6,
        top_p=0.9
    )
    
    # Extract response text generated after the prompt
    generated_output = story_result[0]["generated_text"]
    story_text = generated_output.split("<|assistant|>")[-1].strip()
    
    return story_text


def text2audio(story_text):
    """
    Converts story text to an MP3 audio file using Google Text-to-Speech.
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
