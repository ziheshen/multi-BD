# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export PRETRAINED_BLIP_DIFFUSION="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP-Diffusion/blip-diffusion.tar.gz"
export INSTANCE_DIR="/LAVIS/data/mit_mascot"
export OUTPUT_DIR="./output/with_without_irrel"

python inference.py \
  --finetuned_ckpt="/LAVIS/multi_BlipDisenBooth/output/mit_mascot_bsz_2_with_irrel_and_rel/checkpoint-200/model.safetensors" \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_BLIPdiffusion_name_or_path=$PRETRAINED_BLIP_DIFFUSION \
  --cond_subject="mit_mascot" \
  --tgt_subject="mit_mascot" \
  --resolution=512 \
  --seed="0" \
