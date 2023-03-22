from diffusers import StableDiffusionPipeline
import torch
import os
from diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline

from meddatasets import load_test_data

def inference_basic():
    model_id = "output/DreamBoothDataset4Med_crop_mask_bbox1.5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = lambda images, clip_input: (images, False)
    save_dir = model_id+"/inference_images/"
    os.makedirs(save_dir,exist_ok=True)
    prompt = "a photo of gastroscopy disease"
    for i in range(10):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

        image.save(save_dir+str(i).zfill(5)+".png")
        
def inference_repaint():
    model_id = "/home/qilei/DEVELOPMENT/diffusers/examples/dreambooth/output/DreamBoothDataset4Med_inpaint_crop0_mask1_bbox1.2_db2x5"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = lambda images, clip_input: (images, False)
    
    img_id = 2
    
    save_dir = model_id+"/inference_images/"+str(img_id)+"/"
    os.makedirs(save_dir,exist_ok=True)
    prompt = "a photo of gastroscopy disease"
    
    original_image,mask = load_test_data(img_id,with_crop=False,bbox_extend=1.2)
    size = 512
    original_image = original_image.resize((size,size))
    mask = mask.convert("L").resize((size,size))
    
    for i in range(10):
        image = pipe(prompt, image = original_image, mask_image = mask).images[0]
        image.save(save_dir+str(i).zfill(5)+".png")    
        

if __name__ == '__main__':
    inference_repaint()