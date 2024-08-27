# from diffusers import FluxInpaintPipeline

# MAX_SEED = np.iinfo(np.int32).max
# IMAGE_SIZE = 1024
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", DEVICE)

# pipe = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token="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").to(DEVICE)
from typing import Tuple

import requests
import random
import numpy as np
import gradio as gr
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline

MAX_SEED = np.iinfo(np.int32).max
IMAGE_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)


# pipe = FluxInpaintPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(DEVICE)
pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token="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").to(DEVICE)


def resize_image_dimensions(
    original_resolution_wh: Tuple[int, int],
    maximum_dimension: int = IMAGE_SIZE
) -> Tuple[int, int]:
    width, height = original_resolution_wh

    if width <= maximum_dimension and height <= maximum_dimension:
        width = width - (width % 32)
        height = height - (height % 32)
        return width, height

    if width > height:
        scaling_factor = maximum_dimension / width
    else:
        scaling_factor = maximum_dimension / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    new_width = new_width - (new_width % 32)
    new_height = new_height - (new_height % 32)

    return new_width, new_height


def process_with_flux(
    input_image_editor: dict,
    input_text: str,
    seed_slicer: int,
    randomize_seed_checkbox: bool,
    strength_slider: float,
    num_inference_steps_slider: int,
    progress=gr.Progress(track_tqdm=True)
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

    print("image.size:", image.size)
    width, height = resize_image_dimensions(original_resolution_wh=image.size)
    # resized_image = image.resize((width, height), Image.LANCZOS)
    # resized_mask = mask.resize((width, height), Image.LANCZOS)
    print("width, height:", width, height)
    if randomize_seed_checkbox:
        seed_slicer = random.randint(0, MAX_SEED)
    # print("Seed:", seed_slicer)
    generator = torch.Generator().manual_seed(seed_slicer)
    result = pipe(
        prompt=input_text,
        image=image,
        mask_image=mask,
        width=width,
        height=height,
        strength=strength_slider,
        generator=generator,
        num_inference_steps=num_inference_steps_slider
    ).images[0]
    print('INFERENCE DONE')
    print("result.size:", result.size)
    return result, mask
    # return None, None


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image_editor_component = gr.ImageEditor(
                label='Image',
                type='pil',
                sources=["upload", "webcam"],
                image_mode='RGB',
                layers=False,
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))

            with gr.Row():
                input_text_component = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                submit_button_component = gr.Button(
                    value='Submit', variant='primary', scale=0)

            with gr.Accordion("Advanced Settings", open=False):
                seed_slicer_component = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=42,
                )

                randomize_seed_checkbox_component = gr.Checkbox(
                    label="Randomize seed", value=True)

                with gr.Row():
                    strength_slider_component = gr.Slider(
                        label="Strength",
                        info="Indicates extent to transform the reference `image`. "
                             "Must be between 0 and 1. `image` is used as a starting "
                             "point and more noise is added the higher the `strength`.",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.85,
                    )

                    num_inference_steps_slider_component = gr.Slider(
                        label="Number of inference steps",
                        info="The number of denoising steps. More denoising steps "
                             "usually lead to a higher quality image at the",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=20,
                    )
        with gr.Column():
            output_image_component = gr.Image(
                type='pil', image_mode='RGB', label='Generated image', format="png")
            with gr.Accordion("Debug", open=False):
                output_mask_component = gr.Image(
                    type='pil', image_mode='RGB', label='Input mask', format="png")

    submit_button_component.click(
        fn=process_with_flux,
        inputs=[
            input_image_editor_component,
            input_text_component,
            seed_slicer_component,
            randomize_seed_checkbox_component,
            strength_slider_component,
            num_inference_steps_slider_component
        ],
        outputs=[
            output_image_component,
            output_mask_component
        ]
    )

demo.launch()