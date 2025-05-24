from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import sys
import time

prompt = "an illustration"
sys.path.insert(0, sys.path[0]+"/../")

from lora_diffusion import tune_lora_scale, patch_pipe

torch.manual_seed(time.time())

model_id = "../stable-diffusion/stable-diffusion-1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None

patch_pipe(
    pipe,
     "./output",
    patch_text=True,
    patch_ti=False,
    patch_unet=True,
)

tune_lora_scale(pipe.unet, 0.75)
tune_lora_scale(pipe.text_encoder, 0.75)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7, height=512,width=512).images[0]
image.save("../output/lora_output.jpg")
