#export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
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

export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export OUTPUT_DIR="27_dreambooth_output/DreamBoothDataset4Med_inpaint1_512_crop1_mask1_bbox1.25_db2_6x25/"
accelerate launch train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset="DreamBoothDataset4Med" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=20 --checkpointing_epochs=5 \
  --with_mask \
  --bbox_extend_scale 1.25 \
  --dataset_ids "2,6" \
  --with_crop
  #--max_train_steps=5000

export MODEL_NAME="stabilityai/stable-diffusion-2-inpainting"
export OUTPUT_DIR="27_dreambooth_output/DreamBoothDataset4Med_inpaint2_512_crop1_mask1_bbox1.25_db2_6x25/"
accelerate launch train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset="DreamBoothDataset4Med" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=20 --checkpointing_epochs=5 \
  --with_mask \
  --bbox_extend_scale 1.25 \
  --dataset_ids "2,6" \
  --with_crop

