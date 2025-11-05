import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="AI Background Replacement", page_icon="🎨", layout="centered")
st.title("🎨 AI Background Replacement App")
st.write("Upload your photo and background image, and let AI replace it automatically!")

uploaded_fg = st.file_uploader("📸 Upload a photo with a person:", type=["jpg", "jpeg", "png"])
uploaded_bg = st.file_uploader("🌄 Upload a background image:", type=["jpg", "jpeg", "png"])

if uploaded_fg and uploaded_bg:
    with st.spinner("Processing..."):
        fg_img = Image.open(uploaded_fg).convert("RGB")
        bg_img = Image.open(uploaded_bg).convert("RGB")

        fg_bgr = cv2.cvtColor(np.array(fg_img), cv2.COLOR_RGB2BGR)
        bg_bgr = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2BGR)
        bg_bgr = cv2.resize(bg_bgr, (fg_bgr.shape[1], fg_bgr.shape[0]))

        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segment:
            results = segment.process(cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2RGB))
            mask = results.segmentation_mask

        alpha = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)[..., None]
        alpha = np.clip(alpha, 0.0, 1.0)

        fg_float = fg_bgr.astype(np.float32)
        bg_float = bg_bgr.astype(np.float32)
        output = (alpha * fg_float + (1 - alpha) * bg_float).astype(np.uint8)

        result_img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        st.image(result_img, caption="✅ Background Replaced", use_container_width=True)

        result_pil = Image.fromarray(result_img)
        buf = BytesIO()
        result_pil.save(buf, format="JPEG")
        buf.seek(0)

        st.download_button(
            label="📥 Download Result Image",
            data=buf,
            file_name="output_replaced.jpg",
            mime="image/jpeg"
        )
else:
    st.info("👆 Please upload both images to start.")
