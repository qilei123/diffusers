from diffusers import StableDiffusionPipeline
import torch

model_id = "output/DreamBoothDataset4Med"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "a photo of gastroscopy disease"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("output/DreamBoothDataset4Med/test.png")