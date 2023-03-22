from pycocotools.coco import COCO
import os

gastro_disease_prompt = 'a photo of gastroscopy disease'

dataset_names = ['dataset1','dataset2']

dataset_records = {"dataset1":
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
                }
                }

def load_with_coco_per_ann(root_dir,image_folder='crop_images',
                           ann_file_dir='annotations/instances_default.json', cat_ids=[1]):
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
            instance["bbox"][0] = 1 if instance["bbox"][0]<0 else instance["bbox"][0]
            instance["bbox"][1] = 1 if instance["bbox"][1]<0 else instance["bbox"][1]
            instance["bbox"][2] = img["width"]-instance["bbox"][0] if instance["bbox"][0]+instance["bbox"][2]>img["width"] else instance["bbox"][2]
            instance["bbox"][3] = img["height"]-instance["bbox"][1] if instance["bbox"][1]+instance["bbox"][3]>img["height"] else instance["bbox"][3]
            
            instance_images_path.append(instance["img_dir"])
            instances.append(instance)
            
    return instances, instance_images_path



if __name__ == "__main__":
    pass