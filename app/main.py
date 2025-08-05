# streamlit_app.py — Streamlit frontend
import streamlit as st
from predictor import predict
from PIL import Image
import os
import json

# Load flower facts from JSON file
with open("app/flower_facts.json", "r", encoding="utf-8") as f:
    flower_facts = json.load(f)

st.set_page_config(page_title="Flower Classifier", layout="wide")
#st.title("🌸 Flower Images Classifier")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🌸 Flower Images Classifier</h1>
    """,
    unsafe_allow_html=True
)

# Layout with three columns
left_col, mid_col, right_col = st.columns([1.5, 0.1, 1.5])

with left_col:
    uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        resized_image = image.resize((400, 400))
        st.image(resized_image, caption="Uploaded Image", use_container_width=True)

        # Save image temporarily
        image_path = "temp_upload.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        prediction_ready = True
    else:
        prediction_ready = False

with right_col:
    st.markdown(" ")
    #st.subheader("Prediction")

    if prediction_ready:
        st.markdown(" ")
        try:
            prediction = predict(image_path)
            st.success(f"🌼 The flower in the picture is: **{prediction.title()}**")
            
            info = flower_facts.get(prediction.lower())

            if info:
                st.markdown(f"**About {prediction}:** {info['description']}")
                st.markdown(f"**Key Characteristics:** {info['characteristics']}")
                st.markdown(f"**🌼 Fun Fact:** _{info['fun_fact']}_")
            else:
                st.info("No extra information available for this flower.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
        finally:
            os.remove(image_path)
    else:
        st.markdown(" ")
        st.info("Upload a flower image to get a prediction.")
