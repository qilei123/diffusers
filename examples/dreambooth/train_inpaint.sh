export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="/home/ycao/DEVELOPMENTS/diffusers/datasets/"


# export OUTPUT_DIR="output/DreamBoothDataset4Med_inpaint_512_crop1_mask1_bbox2_db3x15/"
# accelerate launch train_dreambooth_inpaint.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --dataset="DreamBoothDataset4Med" \
#   --resolution=512 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=15 \
#   --with_mask \
#   --bbox_extend_scale 2 \
#   --dataset_id 3 \
#   --with_crop

# export OUTPUT_DIR="output/DreamBoothDataset4Med_inpaint_512_crop1_mask1_bbox2_db4x15/"
# accelerate launch train_dreambooth_inpaint.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --dataset="DreamBoothDataset4Med" \
#   --resolution=512 \
#   --train_batch_size=4 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=15 \
#   --with_mask \
#   --bbox_extend_scale 2 \
#   --dataset_id 4 \
#   --with_crop


export OUTPUT_DIR="output/DreamBoothDataset4Med_inpaint_512_crop1_mask1_bbox2_db3_4_5x10/"
accelerate launch train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset="DreamBoothDataset4Med" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10 \
  --with_mask \
  --bbox_extend_scale 2 \
  --dataset_id 5 \
  --with_crop