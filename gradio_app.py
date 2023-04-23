import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
import requests
from transformers import SamModel, SamProcessor
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def flip_image(inputs):
    dots = inputs['mask']

    gray = cv2.cvtColor(dots, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    kernel = np.ones((5,5),np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for contour in contours:
        moments = cv2.moments(contour)
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        points.append([cx, cy])
    points = [points]
    image_input = inputs['image']
    image_input = Image.fromarray(image_input)
    #points = [[[550, 600], [2100,1000]]]
    inputs = processor(image_input, input_points=points, return_tensors="pt").to(device)
    # Forward pass
    outputs = model(**inputs)

    # Postprocess outputs
    draw = ImageDraw.Draw(image_input)
    for point in points[0]:
        draw.ellipse((point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10), fill="red")


    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    #scores = outputs.iou_scores

    mask = masks[0].squeeze(0).numpy().transpose(1, 2, 0)

    pred_masks = [image_input]
    for i in range(mask.shape[2]):
        #mask[:,:,i] = mask[:,:,i] * scores[0][i].item()
        pred_masks.append(Image.fromarray((mask[:,:,i] * 255).astype(np.uint8)))
    return pred_masks


with gr.Blocks() as demo:
    gr.Markdown("# Demo to run Segment Anything")
    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image(tool='sketch')
            image_output = gr.Gallery()
        
        image_button = gr.Button("Segment Image")

    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()