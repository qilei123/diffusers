from diffusers import StableDiffusionPipeline
import torch
import os
from diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline

from meddatasets import load_test_data,dataset_prompts,dataset_names,load_test_data_coco

import glob

def inference_basic():
    model_id = "output/DreamBoothDataset_NBI_CAT1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = lambda images, clip_input: (images, False)
    save_dir = model_id+"/inference_images/"
    os.makedirs(save_dir,exist_ok=True)
    prompt = "a photo of gastroscopy disease"
    for i in range(10):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

        image.save(save_dir+str(i).zfill(5)+".png")
        
def inference_repaint():
    dataset_id = 5
    model_id = "output/DreamBoothDataset4Med_inpaint_512_crop1_mask1_bbox2_db3_4_"+str(dataset_id)+"x10"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = lambda images, clip_input: (images, False)
    
    original_images,masks = load_test_data_coco(with_crop=True,bbox_extend=2,cat_ids = [1])

    
    for img_id in range(len(original_images)):
        
        save_dir = model_id+"/inference_images/"+str(img_id)+"/"
        
        os.system("rm "+save_dir+"*.png")
        
        os.makedirs(save_dir,exist_ok=True)
        prompt = dataset_prompts[dataset_names[dataset_id]]
        
        original_image,mask = original_images[img_id],masks[img_id]#load_test_data(img_id,with_crop=True,bbox_extend=1.5)
        size = 512
        original_image = original_image.resize((size,size))
        mask = mask.resize((size,size))
        
        for i in range(10):
            image = pipe(prompt, image = original_image, mask_image = mask, num_inference_steps = 50).images[0]
            image.save(save_dir+str(i).zfill(5)+".png")    

def inference_NBI():
    for i in [1,2,3]:
        org_files = glob.glob(os.path.join('/home/ycao/DEVELOPMENTS/diffusers/datasets/nbi_v4/train/',str(i),"*.jpg"))
        generate_n = 2000-len(org_files)
        generate_n = 3000
        model_id = "output/DreamBoothDataset_NBI_CAT"+str(i)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        pipe.safety_checker = lambda images, clip_input: (images, False)
        save_dir = model_id+"/inference_images_a/"
        os.makedirs(save_dir,exist_ok=True)
        prompt = "a photo of gastroscopy disease"
        for i in range(generate_n):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

            image.save(save_dir+str(i).zfill(5)+".png")

if __name__ == '__main__':
    #inference_repaint()
    #inference_basic()
    inference_NBI()
    pass