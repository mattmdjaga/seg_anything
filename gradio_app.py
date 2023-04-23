import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
import requests
from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
#model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
#processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def flip_image(image_input):
    image_input = Image.fromarray(image_input)
    points = [[[550, 600], [2100,1000]]]
    draw = ImageDraw.Draw(image_input)
    print(image_input.size)
    for point in points[0]:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill="red")
    return image_input


with gr.Blocks() as demo:
    gr.Markdown("# Demo to run Segment Anything")
    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        
        image_button = gr.Button("Segment Image")

    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()