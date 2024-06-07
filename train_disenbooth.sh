# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export PRETRAINED_BLIP_DIFFUSION="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP-Diffusion/blip-diffusion.tar.gz"
export INSTANCE_DIR="/LAVIS/data/mit_mascot"
export OUTPUT_DIR="./output/test"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_BLIPdiffusion_name_or_path=$PRETRAINED_BLIP_DIFFUSION \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --subject_text="a mit_mascot" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=200 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_prompt="A mit_mascot in the jungle" \
  --validation_epochs=200 \
  --seed="0" \
