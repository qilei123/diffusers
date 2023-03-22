export MODEL_NAME="runwayml/stable-diffusion-inpainting"
export INSTANCE_DIR="/home/qilei/DEVELOPMENT/diffusers/datasets/"
export OUTPUT_DIR="output/DreamBoothDataset4Med_crop_mask_bbox1.5/"

accelerate launch train_dreambooth_inpaint.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset="DreamBoothDataset4Med" \
  --instance_prompt="a photo of gastroscopy disease" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --with_mask