# from diffusers import FluxInpaintPipeline

# MAX_SEED = np.iinfo(np.int32).max
# IMAGE_SIZE = 1024
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", DEVICE)

# pipe = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token="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").to(DEVICE)
from typing import Tuple, Optional

import cv2
import random
import numpy as np
import gradio as gr
import torch
from PIL import Image, ImageFilter
from diffusers import FluxInpaintPipeline

MAX_SEED = np.iinfo(np.int32).max
IMAGE_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pipe = FluxInpaintPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(DEVICE)
pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token="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").to(DEVICE)

def calculate_image_dimensions_for_flux(
    original_resolution_wh: Tuple[int, int],
    maximum_dimension: int = IMAGE_SIZE
) -> Tuple[int, int]:
    width, height = original_resolution_wh

    if width > height:
        scaling_factor = maximum_dimension / width
    else:
        scaling_factor = maximum_dimension / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    new_width = new_width - (new_width % 32)
    new_height = new_height - (new_height % 32)

    return new_width, new_height

def process_mask(
    mask: Image.Image,
    mask_inflation: Optional[int] = None,
    mask_blur: Optional[int] = None
) -> Image.Image:
    """
    Inflates and blurs the white regions of a mask.
    Args:
        mask (Image.Image): The input mask image.
        mask_inflation (Optional[int]): The number of pixels to inflate the mask by.
        mask_blur (Optional[int]): The radius of the Gaussian blur to apply.
    Returns:
        Image.Image: The processed mask with inflated and/or blurred regions.
    """
    if mask_inflation and mask_inflation > 0:
        mask_array = np.array(mask)
        kernel = np.ones((mask_inflation, mask_inflation), np.uint8)
        mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        mask = Image.fromarray(mask_array)

    if mask_blur and mask_blur > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))

    return mask

def process_with_flux(
    input_image_editor: dict,
    input_text: str,
    seed_slicer: int,
    randomize_seed_checkbox: bool,
    strength_slider: float,
    num_inference_steps_slider: int
):
    if not input_text:
        gr.Info("Please enter a text prompt.")
        return None, None
    image = input_image_editor['background']
    mask = input_image_editor['layers'][0]

    if not image:
        gr.Info("Please upload an image.")
        return None, None

    if not mask:
        gr.Info("Please draw a mask on the image.")
        return None, None

    width, height = calculate_image_dimensions_for_flux(original_resolution_wh=image.size)
    image = image.resize((width, height), Image.LANCZOS)
    mask = mask.resize((width, height), Image.LANCZOS)
    mask = process_mask(mask, mask_inflation=5, mask_blur=5)
    if randomize_seed_checkbox:
        seed_slicer = random.randint(0, MAX_SEED)
    
    generator = torch.Generator("cpu").manual_seed(seed_slicer)
    result = pipe(
        prompt=input_text,
        image=image,
        mask_image=mask,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps_slider,
        max_sequence_length=512,
        strength=strength_slider,
        generator=generator
    ).images[0]
    return result
    # return None, None


# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column():
#             input_image_editor_component = gr.ImageEditor(
#                 label='Image',
#                 type='pil',
#                 sources=["upload", "webcam"],
#                 image_mode='RGB',
#                 layers=False,
#                 brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))

#             with gr.Row():
#                 input_text_component = gr.Text(
#                     label="Prompt",
#                     show_label=False,
#                     max_lines=1,
#                     placeholder="Enter your prompt",
#                     container=False,
#                 )
#                 submit_button_component = gr.Button(
#                     value='Submit', variant='primary', scale=0)

#             with gr.Accordion("Advanced Settings", open=False):
#                 seed_slicer_component = gr.Slider(
#                     label="Seed",
#                     minimum=0,
#                     maximum=MAX_SEED,
#                     step=1,
#                     value=42,
#                 )

#                 randomize_seed_checkbox_component = gr.Checkbox(
#                     label="Randomize seed", value=True)

#                 with gr.Row():
#                     strength_slider_component = gr.Slider(
#                         label="Strength",
#                         info="Indicates extent to transform the reference `image`. "
#                              "Must be between 0 and 1. `image` is used as a starting "
#                              "point and more noise is added the higher the `strength`.",
#                         minimum=0,
#                         maximum=1,
#                         step=0.01,
#                         value=0.85,
#                     )

#                     num_inference_steps_slider_component = gr.Slider(
#                         label="Number of inference steps",
#                         info="The number of denoising steps. More denoising steps "
#                              "usually lead to a higher quality image at the",
#                         minimum=1,
#                         maximum=50,
#                         step=1,
#                         value=20,
#                     )
#         with gr.Column():
#             output_image_component = gr.Image(
#                 type='pil', image_mode='RGB', label='Generated image', format="png")
#             with gr.Accordion("Debug", open=False):
#                 output_mask_component = gr.Image(
#                     type='pil', image_mode='RGB', label='Input mask', format="png")

#     submit_button_component.click(
#         fn=process_with_flux,
#         inputs=[
#             input_image_editor_component,
#             input_text_component,
#             seed_slicer_component,
#             randomize_seed_checkbox_component,
#             strength_slider_component,
#             num_inference_steps_slider_component
#         ],
#         outputs=[
#             output_image_component,
#             output_mask_component
#         ]
#     )

# demo.launch()