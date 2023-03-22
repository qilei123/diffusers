from diffusers import StableDiffusionPipeline
import torch
import os

model_id = "output/DreamBoothDataset4Med"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
save_dir = "output/DreamBoothDataset4Med/inference_images/"
os.makedirs(save_dir,exist_ok=True)
prompt = "a photo of gastroscopy disease"
for i in range(100):
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    image.save(save_dir+str(i).zfill(5)+".png")