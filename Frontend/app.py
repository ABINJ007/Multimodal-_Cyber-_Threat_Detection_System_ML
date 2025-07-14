import streamlit as st
import requests
import time

# Set page title and layout
st.set_page_config(page_title="All Safe - Multimodal Threat Detection", layout="centered")

# Logo and name
st.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="font-size: 50px; margin-bottom: 5px;">🛡️ All Safe</h1>
        <p style="font-size: 18px; color: #555;">AI-Based Multimodal Threat Detection Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Description
st.markdown(
    """
    This system helps detect potential threats across different modalities like **URLs** and **Images** using AI.
    """
)

# Input Type
option = st.selectbox("Choose input type:", ["URL", "Image"])

# URL Threat Detection
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
                response = requests.post("http://127.0.0.1:8000/scan", json={"url": url}, timeout=10)
                response.raise_for_status()
                result = response.json()

                # Debug: Print keys from the response
                st.write("Response keys:", result.keys())

                mse = result.get('mse', None)
                threshold = result.get('threshold', None)

                if result.get("threat"):
                    st.error(f"🚨 Fake URL Detected: {result.get('message', 'No message provided')}")
                else:
                    st.success(f"✅ Safe URL: {result.get('message', 'No message provided')}")
                    st.balloons()

                if mse is not None and threshold is not None:
                    st.write(f"MSE: {mse:.6f}, Threshold: {threshold:.6f}")
                else:
                    st.warning("MSE or threshold info not available in the response.")

            except requests.exceptions.RequestException as e:
                st.error("Failed to connect to backend or process the request.")
                st.write(str(e))

# Image Threat Detection Placeholder
elif option == "Image":
    image_file = st.file_uploader("Upload an image (e.g., meme, screenshot):", type=["jpg", "jpeg", "png"])

    if st.button("Scan Image"):
        if image_file:
            st.image(image_file, caption="Uploaded Image", use_column_width=True)
            with st.spinner("Processing image..."):
                time.sleep(2)
            st.info("🧠 This is a placeholder. Image threat detection coming soon.")
        else:
            st.warning("⚠️ Please upload an image.")
