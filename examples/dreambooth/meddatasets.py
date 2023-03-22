from pycocotools.coco import COCO
import os
from skimage import draw
import numpy as np
from PIL import Image
from PIL import ImageFilter
import torch
import cv2
import json
from math import floor
from scipy import ndimage
import glob

gastro_disease_prompt = 'a photo of gastroscopy disease'

dataset_names = ['dataset_test','dataset1','dataset2','polyp1']

dataset_records = {"dataset_test":{"gastro_cancer/xiehe_far_1":[1],},
                "dataset1":
                {"gastro_cancer/xiehe_far_1":[1], #0
                "gastro_cancer/xiehe_far_2":[1], #1
                "gastro_cancer/xiangya_far_2021":[1], #2
                "gastro_cancer/xiangya_far_2022":[1], #3
                },
                "dataset2":
                {"gastro_cancer/xiehe_far_1":[1], #0
                "gastro_cancer/xiehe_far_2":[1], #1
                "gastro_cancer/xiangya_far_2021":[1], #2
                "gastro_cancer/xiangya_far_2022":[1], #3
                "gastro_cancer/gastro8-12/2021-2022年癌变已标注/20221111/2021_2022_癌变_20221111":[1,4,5], #4
                "gastro_cancer/gastro8-12/低级别_2021_2022已标注/2021_2022_低级别_20221110":[1,4,5], #5
                "gastro_cancer/gastro8-12/协和2022_第一批胃早癌视频裁图已标注/20221115/癌变2022_20221115":[1,4,5], #6
                "gastro_cancer/gastro8-12/协和2022_第二批胃早癌视频裁图已标注/协和_2022_癌变_2_20221117":[1,4,5], #7
                "gastro_cancer/gastro8-12/协和21-11月~2022-5癌变已标注/协和2021-11月_2022-5癌变_20221121":[1,4,5], #8
                },
                "polyp1":
                    {"polyp/huiwei_dataset_fuji_additional_V1":[1],}
                }

def extend_bbox(bbox,bbox_extend_scale):
    bbox[0] -= bbox[2]*(bbox_extend_scale-1)/2
    bbox[1] -= bbox[3]*(bbox_extend_scale-1)/2
    bbox[2] = bbox[2]*bbox_extend_scale
    bbox[3] = bbox[3]*bbox_extend_scale
    
    return [int(value) for value in bbox]

def check_and_fix_bbox(bbox,width,height):
    bbox[0] = 1 if bbox[0]<0 else bbox[0]
    bbox[1] = 1 if bbox[1]<0 else bbox[1]
    bbox[2] = width-bbox[0] if bbox[0]+bbox[2]>width else bbox[2]
    bbox[3] = height-bbox[1] if bbox[1]+bbox[3]>height else bbox[3]    
    return [int(value) for value in bbox] 

def load_with_coco_per_ann(root_dir,image_folder='crop_images',
                           ann_file_dir='annotations/crop_instances_default.json', cat_ids=[1]):
    coco = COCO(os.path.join(root_dir,ann_file_dir))
    
    instance_images_path = []
    instances = []
    
    #以每个病变为单例进行输入
    for ann_key in coco.anns:
        ann = coco.anns[ann_key]
        instance = {}
        if len(ann["segmentation"])>0 and len(ann["segmentation"][0])>0 and (ann['category_id'] in cat_ids):#这里默认每个目标只有一个segmentation标注
            instance["polygon"] = ann["segmentation"][0]
            instance["bbox"] = ann["bbox"]
                
            instance["cat_id"] = ann["category_id"]
            
            img = coco.loadImgs([ann["image_id"]])[0]
            instance["img_dir"] = os.path.join(root_dir,image_folder,img['file_name'])
            instance["img_shape"] = [img['width'],img['height']]
            instance["img_width"],instance["img_height"] = img['width'],img['height']
            
            #check the boundary of the bbox
            instance["bbox"] = check_and_fix_bbox(instance["bbox"],img["width"],img["height"])
            #instance["bbox"][0] = 1 if instance["bbox"][0]<0 else instance["bbox"][0]
            #instance["bbox"][1] = 1 if instance["bbox"][1]<0 else instance["bbox"][1]
            #instance["bbox"][2] = img["width"]-instance["bbox"][0] if instance["bbox"][0]+instance["bbox"][2]>img["width"] else instance["bbox"][2]
            #instance["bbox"][3] = img["height"]-instance["bbox"][1] if instance["bbox"][1]+instance["bbox"][3]>img["height"] else instance["bbox"][3]
            
            instance_images_path.append(instance["img_dir"])
            instances.append(instance)
            
    return instances, instance_images_path

def load_polyp(root_dir,images_folder='',anns_folder='annotations',cat_ids = [1,]):
    
    annotation_dir_list = glob.glob(os.path.join(root_dir,anns_folder,"*.json"))
    annotation_file_list = [os.path.basename(annotation_dir) for annotation_dir in annotation_dir_list]
    
    instances, instance_images_path = [],[]
    
    for annotation_file in annotation_file_list:
        temp_instances,temp_instance_images_path = load_with_coco_per_ann(root_dir,images_folder,os.path.join(anns_folder,annotation_file),cat_ids=cat_ids)
        instances += temp_instances
        instance_images_path += temp_instance_images_path
        
    return instances,instance_images_path

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def polygon2vertex_coords(polygon):
    col_coords = polygon[::2]
    row_coords = polygon[1::2]
    return [row_coords,col_coords]

def preprocess(instance_image_path,instance,with_crop = False,bbox_extend=1):
    image = Image.open(instance_image_path)
    if bbox_extend>1:
        #instance["bbox"][0] -= instance["bbox"][2]*(bbox_extend-1)/2
        #instance["bbox"][1] -= instance["bbox"][3]*(bbox_extend-1)/2
        #instance["bbox"][2] = instance["bbox"][2]*bbox_extend
        #instance["bbox"][3] = instance["bbox"][3]*bbox_extend 
        instance["bbox"] = extend_bbox(instance["bbox"],bbox_extend)

    #check the boundary of the bbox
    instance["bbox"][0] = 1 if instance["bbox"][0]<0 else instance["bbox"][0]
    instance["bbox"][1] = 1 if instance["bbox"][1]<0 else instance["bbox"][1]
    instance["bbox"][2] = instance["img_width"]-instance["bbox"][0] if instance["bbox"][0]+instance["bbox"][2]>instance["img_width"] else instance["bbox"][2]
    instance["bbox"][3] = instance["img_height"]-instance["bbox"][1] if instance["bbox"][1]+instance["bbox"][3]>instance["img_height"] else instance["bbox"][3]
     
    
    if with_crop:
        image = image.crop((int(instance['bbox'][0]),int(instance['bbox'][1]),
                            int(instance['bbox'][0]+instance['bbox'][2]),int(instance['bbox'][1]+instance['bbox'][3])))                
    
    mask = poly2mask(*polygon2vertex_coords(instance['polygon']),(instance['img_shape'][1],instance['img_shape'][0]))
    
    if with_crop:
        mask = mask[int(instance['bbox'][1]):int(instance['bbox'][1]+instance['bbox'][3]),
                    int(instance['bbox'][0]):int(instance['bbox'][0]+instance['bbox'][2])]


def get_test_samples(preprocess=None,with_crop=True,blur_mask = False,
                     dynamic_blur_mask = False,blur_kernel_scale = 10,
                     bbox_extend=1.5,stack = False):
    #todo:这里需要将四张测试图片加载进来，并且需要进行preprocess
    data_folder = '/home/qilei/DEVELOPMENT/diffusers/examples/dreambooth/test_med_data'
    test_images_record = open(os.path.join(data_folder,'choose_test_gastro_images.txt'))
    
    records = []
    
    line = test_images_record.readline()
    
    #bbox_extend-=1
    
    while line:
        
        records.append(line[:-1])
        
        line = test_images_record.readline()
        
    image_list = records[::2]
    ann_list = records[1::2]
    
    sample_images = []
    sample_masks = []
    
    for image_name,ann in zip(image_list,ann_list):
        ann = json.loads(ann)
        
        if bbox_extend>1:
            ann["bbox"][0] -= ann["bbox"][2]*(bbox_extend-1)/2
            ann["bbox"][1] -= ann["bbox"][3]*(bbox_extend-1)/2
            ann["bbox"][2] = ann["bbox"][2]*bbox_extend
            ann["bbox"][3] = ann["bbox"][3]*bbox_extend
        
        sample = {}
        
        image = Image.open(os.path.join(data_folder,image_name))
        
        height = image.height
        
        width = image.width
        
        ann["bbox"][0] = 1 if ann["bbox"][0]<0 else ann["bbox"][0]
        ann["bbox"][1] = 1 if ann["bbox"][1]<0 else ann["bbox"][1]
        ann["bbox"][2] = width-ann["bbox"][0] if ann["bbox"][0]+ann["bbox"][2]>width else ann["bbox"][2]
        ann["bbox"][3] = height-ann["bbox"][1] if ann["bbox"][1]+ann["bbox"][3]>height else ann["bbox"][3]        
        
        if with_crop:
            image = image.crop((int(ann['bbox'][0]),int(ann['bbox'][1]),
                        int(ann['bbox'][0]+ann['bbox'][2]),int(ann['bbox'][1]+ann['bbox'][3])))
        
        sample["image"] = image

        mask = poly2mask(*polygon2vertex_coords(ann["segmentation"][0]),(height,width))
        
        #添加掩码边缘高斯模糊的操作
        if dynamic_blur_mask:
            min_box_edge = min(ann['bbox'][2],ann['bbox'][3])
            ksize = floor(min_box_edge/blur_kernel_scale)+1-floor(min_box_edge/blur_kernel_scale)%2
        else:
            ksize = 15
        
        if blur_mask:
            mask = cv2.GaussianBlur(mask, (ksize,ksize), 0)
        
        if with_crop:
            mask = mask[int(ann['bbox'][1]):int(ann['bbox'][1]+ann['bbox'][3]),
                        int(ann['bbox'][0]):int(ann['bbox'][0]+ann['bbox'][2])]   
        sample["mask"]= mask
  
        if preprocess:
            image_tensor = preprocess[0](sample['image']) #totensor
            mask_tensor = preprocess[0](sample['mask']) #totensor
            #mask_tensor = torch.stack((mask_tensor,mask_tensor,mask_tensor),dim=1)
            mask_tensor = mask_tensor.repeat(3,1,1)
            
            image_mask_stack = torch.cat((image_tensor,mask_tensor)).to(torch.float32) 
            
            #for i in range(1,len(preprocess)-1): #不需要再做flip
            image_mask_stack = preprocess[1](image_mask_stack)#scale操作
            #sample['image'] = self.transforms[:-1](sample['image'])
            sample['image'],sample['mask'] = torch.split(image_mask_stack, 3)
            sample['image'] = preprocess[4](sample["image"])  #只在image上做normalize 
            
        sample_images.append(sample['image'])
        sample_masks.append(sample["mask"])
    if stack:    
        images = torch.stack(sample_images)
        masks = torch.stack(sample_masks)
            
        return images,masks  
    else:
        return sample_images,sample_masks
def load_test_data(instance_index=0,with_crop = False,bbox_extend = 1):
    images,masks = get_test_samples(with_crop = with_crop,bbox_extend=bbox_extend)
    
    assert len(images)>instance_index, "no more than "+str(instance_index+1)+" test images"
    
    return images[instance_index],Image.fromarray(masks[instance_index]*255)


def patch_on_original_image(patch_direct = True,bbox_extend = 1.2):
    
    test_images_record = open('test_med_data/choose_test_gastro_images.txt')
    
    patchs_dir = "output/DreamBoothDataset4Med_inpaint_crop1_mask1_bbox1.2_db2x5/inference_images/"
    
    image_names= ["00005.png","00008.png","00008.png","00001.png"]
    
    patch_images = []

    for idx, image_name in enumerate(image_names):
        patch_images.append(Image.open(os.path.join(patchs_dir,str(idx),image_name)))   
    
    records = []
    
    line = test_images_record.readline()
    
    #bbox_extend-=1
    
    while line:
        
        records.append(line[:-1])
        
        line = test_images_record.readline()
        
    image_list = records[::2]
    ann_list = records[1::2]
    
    os.makedirs(os.path.join(patchs_dir,"patched_images"),exist_ok=True)
    
    for image_name,ann,patch in zip(image_list,ann_list,patch_images):
        
        ann = json.loads(ann)    
        image = Image.open(os.path.join("test_med_data",image_name))

        mask = poly2mask(*polygon2vertex_coords(ann["segmentation"][0]),(image.height,image.width))
        
        actual_bbox = [int(ann['bbox'][0]-ann['bbox'][2]*(bbox_extend-1)/2),
                       int(ann['bbox'][1]-ann['bbox'][3]*(bbox_extend-1)/2),
                       int(ann['bbox'][2]*bbox_extend),
                       int(ann['bbox'][3]*bbox_extend)]
        
        actual_bbox[0] = 1 if actual_bbox[0]<0 else actual_bbox[0]
        actual_bbox[1] = 1 if actual_bbox[1]<0 else actual_bbox[1]
        actual_bbox[2] = image.width-actual_bbox[0] if actual_bbox[0]+actual_bbox[2]>image.width else actual_bbox[2]
        actual_bbox[3] = image.height-actual_bbox[1] if actual_bbox[1]+actual_bbox[3]>image.height else actual_bbox[3]           
        
        
        patch = patch.resize((int(actual_bbox[2]),int(actual_bbox[3])))
        if patch_direct:
            image.paste(patch, (int(actual_bbox[0]),int(actual_bbox[1])))
        else:
            
            patch_image = Image.new("RGB",image.size,(0,0,0))
            patch_image.paste(patch, (int(actual_bbox[0]),int(actual_bbox[1])))
            
            mask = ndimage.binary_dilation(mask,iterations=10).astype(mask.dtype)
            #mask = ndimage.gaussian_filter(mask, sigma=0,radius=5)
            
            mask = Image.fromarray(mask).convert('L')
            mask = mask.filter(ImageFilter.GaussianBlur(10))
            
            mask = np.array(mask)
            
            mask1 = mask.copy()
            mask1[int(actual_bbox[1]):int(actual_bbox[1]+actual_bbox[3]),int(actual_bbox[0]):int(actual_bbox[0]+actual_bbox[2])] =0
            mask = mask-mask1
            
            mask = Image.fromarray(mask*255*0.8).convert('L')
            
            image.paste(patch_image, (0,0),mask)
        
        image.save(os.path.join(patchs_dir,"patched_images",image_name))

if __name__ == "__main__":
    #patch_on_original_image(False)
    o1,o2 = load_polyp("/home/qilei/DEVELOPMENT/diffusers/datasets/polyp/huiwei_dataset_fuji_additional_V1")
    print(len(o2))
    