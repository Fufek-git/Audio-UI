import numpy as np
import cv2
import tempfile
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import AudioLDMPipeline
import torch

rate = 16000
neg = "low quality, average quality"
scale = 3


repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

# This script generates sound (first tab)
def generate_sound(input, steps, length, beams):
    # Convert the Numpy array to an image
    img = input.astype(np.uint8)

# Save the image as a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
        temp_path = temp_img.name
        cv2.imwrite(temp_path, img)
        raw_image = img  # Pass the path of the image instead of a list

    # Conditional image captioning
    text = "sound of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    prompt = caption
    print(prompt)
    audio = pipe(prompt, negative_prompt=neg, num_inference_steps=steps, audio_length_in_s=length, num_waveforms_per_prompt=beams, guidance_scale=scale).audios[0]
    return (rate, audio)  # Return the audio


# This script generates music (second tab)
def generate_music(input, steps, length, beams):
    # Convert the NumPy array to an image
    img = input.astype(np.uint8)

# Save the image as a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_img:
        temp_path = temp_img.name
        cv2.imwrite(temp_path, img)
        raw_image = img  # Pass the path of the image instead of a list

    text = "music of"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True) 
    prompt = caption
    print(prompt)
    audio = pipe(prompt, negative_prompt=neg, num_inference_steps=steps, audio_length_in_s=length, num_waveforms_per_prompt=beams, guidance_scale=scale).audios[0]
    return (rate, audio)  # Return the audio

# Script for generating music (third tab)
def generate_music1(input, steps, length, beams):
    print(str(input))
    audio = pipe(str(input), negative_prompt=neg, num_inference_steps=steps, audio_length_in_s=length, num_waveforms_per_prompt=beams, guidance_scale=scale).audios[0]
    return (rate, audio)
# Gui settings
with gr.Blocks() as demo:
    gr.Markdown("Generate sound using this gradio app.")
    with gr.Tab("Image to Sound Generation"):
        with gr.Row():
            image_input = gr.Image()
        text_button = gr.Button("Generate Sound Effect")    
        audio_output = gr.Audio()
    with gr.Tab("Image to Music Generation"):
        with gr.Row():
            image_input1 = gr.Image()
        text_button1 = gr.Button("Generate Music")    
        audio_output1 = gr.Audio()
    with gr.Tab("Text to Music Generation"):
        with gr.Row():
            text_input = gr.Textbox(label="Prompt:")
        text_button2 = gr.Button("Generate Music")    
        audio_output2 = gr.Audio()
    # Settings
    examples=["train.jpg", "walking.jpg"]
    with gr.Accordion("Settings:", open=False):
        steps = gr.Slider(5, 100, value=50, step=5, label="Steps:", info="Number of denoising cycles to generate the audio" )
        length = gr.Slider(5, 160, value=10, step=5, label="Length:", info="Length of the generated audio in seconds")
        beams = gr.Slider(1, 10, value=3, step=1, label="Number of waveforms:", info="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation")
    # Functionality    
    text_button.click(generate_sound, inputs=[image_input, steps, length, beams], outputs=audio_output)
    text_button1.click(generate_music, inputs=[image_input1, steps, length, beams], outputs=audio_output1)
    text_button2.click(generate_music1, inputs=[text_input, steps, length, beams], outputs=audio_output2)

demo.launch()
