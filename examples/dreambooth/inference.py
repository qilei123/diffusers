from diffusers import StableDiffusionPipeline
import torch
import os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline
from diffusers.utils import load_image,torch_device
from diffusers import PaintByExamplePipeline

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
    dataset_id = 0
    #model_id = "output/DreamBoothDataset4Med_inpaint_512_crop1_mask1_bbox1.2_db0x100"
    model_id = "27_dreambooth_output/DreamBoothDataset4Med_inpaint2_512_crop1_mask1_bbox1.25_db2_6x25"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = lambda images, clip_input: (images, False)
    
    original_images,masks = load_test_data_coco(with_crop=True,bbox_extend=1.25,cat_ids = [1,2])

    
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
    for i in [3]:
        org_files = glob.glob(os.path.join('/home/ycao/DEVELOPMENTS/diffusers/datasets/nbi_v4/train/',str(i),"*.jpg"))
        generate_n = 5000-len(org_files)
        #generate_n = 3000
        model_id = "output/DreamBoothDataset_NBI_CAT"+str(i)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
        pipe.safety_checker = lambda images, clip_input: (images, False)
        save_dir = model_id+"/inference_images_a/"
        os.makedirs(save_dir,exist_ok=True)
        prompt = "a photo of gastroscopy disease"
        for i in range(generate_n):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

            image.save(save_dir+str(i).zfill(5)+".png")

def inference_repaint_random_mask():
    pass

current_folder = 'examples/dreambooth/'

def load_selected_patches():
    patches_folder = current_folder + '27_dreambooth_output/DreamBoothDataset4Med_inpaint_512_crop1_mask1_bbox1.25_db2_3x20/inference_images'
    patches_num = 13
    patches_ids = [4]*patches_num
    patches = []
    for i in range(patches_num):
        patches.append(Image.open(os.path.join(patches_folder,str(i),str(patches_ids[i]).zfill(5)+'.png')))
        
    return patches
def test_paint_by_example():
    original_images,masks = load_test_data_coco(with_crop=False,bbox_extend=1.25,cat_ids = [1,2])
    patches = load_selected_patches()
    # make sure here that pndm scheduler skips prk
    '''
    init_image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        "/paint_by_example/dog_in_bucket.png"
    )
    mask_image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        "/paint_by_example/mask.png"
    )
    example_image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        "/paint_by_example/panda.jpg"
    )
    '''
    input_size = 512
    pipe = PaintByExamplePipeline.from_pretrained("Fantasy-Studio/Paint-by-Example")
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    index = 0
    for init_image, mask_image,example_image in zip(original_images,masks,patches):
        init_image = init_image.resize((input_size,input_size))
        #mask_image = mask_image*255
        mask_image = mask_image.convert('RGB').resize((input_size,input_size))
        generator = torch.manual_seed(321)
        output = pipe(
            image=init_image,
            mask_image=mask_image,
            example_image=example_image,
            generator=generator,
            guidance_scale=5.0,
            num_inference_steps=50,
        )

        image = output.images
        
        temp_dir = current_folder + "27_dreambooth_output/TEMP/"
        
        init_image.save(temp_dir+str(index)+"_org_test_pbe.png")
        mask_image.save(temp_dir+str(index)+"_pbe_mask.png")
        example_image.save(temp_dir+str(index)+"_pbe_example.png")
        image[0].save(temp_dir+str(index)+"_test_pbe.png")
        index+=1


if __name__ == '__main__':
    inference_repaint()
    #inference_basic()
    #inference_NBI()
    #test_paint_by_example()
    pass