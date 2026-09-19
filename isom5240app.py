import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Age Classification using ViT", layout="centered")

st.title("Age Classification using ViT")
st.write("Upload an image to predict the age range using Vision Transformer.")


def load_model():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")


age_classifier = load_model()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying image..."):
        age_predictions = age_classifier(image)
        age_predictions = sorted(
            age_predictions, key=lambda x: x["score"], reverse=True
        )

    st.subheader("Predicted Age Range:")
    top_prediction = age_predictions[0]
    st.success(
        f"**Age range:** {top_prediction['label']} ({top_prediction['score']*100:.1f}% confidence)"
    )

    with st.expander("Show all class probabilities"):
        for pred in age_predictions:
            st.write(f"**{pred['label']}**: {pred['score']*100:.2f}%")
