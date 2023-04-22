import streamlit as st
import numpy as np
import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="Segment Anything")
st.write("## Get a mask from a point in an image")
st.write("""
    This app uses the [Segment Anything](https://huggingface.co/facebook/sam-vit-base)
    model from Meta to get a mask from a point in an image.
    """)

st.sidebar.write("## Upload and download :gear:")

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = image.resize((image.width // 4, image.height // 4))
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    fix_image(upload=my_upload)
else:
    fix_image("car.png")

'''

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Load image
url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
image = Image.open(requests.get(url, stream=True).raw)

# Preprocess image by making it 4x smaller
image = image.resize((image.width // 4, image.height // 4))

points = [[[115, 150]]]

inputs = processor(image, input_points=points, return_tensors="pt").to(device)

# Forward pass
outputs = model(**inputs)

# Postprocess outputs


masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores

mask = masks[0].squeeze(0).numpy().transpose(1, 2, 0)

'''