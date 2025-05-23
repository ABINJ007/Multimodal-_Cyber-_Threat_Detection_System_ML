import streamlit as st
import requests
import time

st.set_page_config(page_title="Multimodal Threat Detection", layout="centered")
st.title("AI-Based Multimodal Threat Detection Platform")

st.markdown("This system helps detect potential threats across different modalities like **URLs**, **Text**, **Images**, and **Audio** using AI.")

option = st.selectbox("Choose input type:", ["URL", "Text", "Image", "Audio"])

if option == "URL":
    url = st.text_input("Enter URL to scan:")
    
    if st.button("Scan URL"):
        if not url.strip():
            st.warning("⚠️ Please enter a valid URL before scanning.")
        else:
            with st.spinner("Analyzing the URL, please wait..."):
                progress_bar = st.progress(0)
                for percent in range(0, 101, 20):
                    time.sleep(0.3)
                    progress_bar.progress(percent)

            try:
                response = requests.post("http://127.0.0.1:8000/scan", json={"url": url})
                result = response.json()
                if result["threat"]:
                    st.error(f"🚨 Threat detected! {result['message']}")
                else:
                    st.success(f"✅ {result['message']}")
                    st.balloons()
            except Exception as e:
                st.error("❌ Failed to connect to backend.")
                st.write(e)

elif option == "Text":
    text_input = st.text_area("Paste message or post to analyze:")
    
    if st.button("Analyze Text"):
        with st.spinner("Analyzing text for threats..."):
            time.sleep(2)
        st.info("This is a placeholder. Text analysis module coming soon.")


elif option == "Image":
    image_file = st.file_uploader("Upload an image (e.g., meme, screenshot):", type=["jpg", "jpeg", "png"])
    
    if st.button("Scan Image"):
        if image_file:
            st.image(image_file, caption="Uploaded Image", use_column_width=True)
            with st.spinner("Processing image..."):
                time.sleep(2)
            st.info("This is a placeholder. Image threat detection coming soon.")
        else:
            st.warning("Please upload an image.")

elif option == "Audio":
    audio_file = st.file_uploader("Upload an audio file (e.g., speech, alarm):", type=["wav", "mp3", "ogg"])
    
    if st.button("Scan Audio"):
        if audio_file:
            st.audio(audio_file)
            with st.spinner("Analyzing audio..."):
                time.sleep(2)
            st.info("This is a placeholder. Audio analysis module coming soon.")
        else:
            st.warning("Please upload an audio file.")
