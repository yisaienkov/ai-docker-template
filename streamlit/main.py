import sys
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from loguru import logger
from PIL.PngImagePlugin import PngImageFile
from streamlit.delta_generator import DeltaGenerator


title = "Image segmentation"
st.markdown(
    f"<h1 style='text-align: center; color: red;'> " f"{title} </h1>", unsafe_allow_html=True
)

logger.configure(
    handlers=[
        {"sink": sys.stderr, "level": "DEBUG"},
        dict(
            sink="logs/debug.log",
            format="{time} {level} {message}",
            level="DEBUG",
            rotation="1 weeks",
        ),
    ]
)


def bytes_to_image(file: bytes, log=None) -> PngImageFile:
    bytes_data = file.getvalue()
    np_array = np.fromstring(bytes_data, np.uint8)

    image_array = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    if log:
        log.info(f"image.shape = {image_array.shape}")

    return Image.fromarray(image_array)


def vis_image(image: PngImageFile, col: DeltaGenerator, text: str) -> None:
    with col:
        st.image(image, width=350)


if __name__ == "__main__":
    uploaded_file = st.file_uploader("", type="jpg")
    before_col, after_col = st.columns(2)

    if uploaded_file is not None:
        logger.info("Load image")

        input_image = bytes_to_image(uploaded_file, logger)
        vis_image(input_image, before_col, "Before")

        out_img = BytesIO()
        input_image.save(out_img, format='PNG')
        out_img.seek(0) 
        output_image = bytes_to_image(out_img, logger)
        
        vis_image(output_image, after_col, "After")
