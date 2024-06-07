
import copy
import gc
import hashlib
import itertools
import logging
import math
import os
import shutil
import warnings
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from disen_net import Image_adapter, cal_cos
import open_clip
import torch.nn as nn


import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
# from diffusers.loaders import (
#     LoraLoaderMixin,
#     text_encoder_lora_state_dict,
# )
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
# from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import unet_lora_state_dict
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_diffusion_models.modeling_ctx_clip import CtxCLIPTextModel
from multi_BD import MultiBlipDisenBooth
from transformers.activations import QuickGELUActivation as QuickGELU
from utils import parse_args

logger = get_logger(__name__)

def train(args):

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.logging_dir is not None:
            os.makedirs(args.logging_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )
    
    # Make one log on every process with the configuration for debugging.
    t = time.localtime()
    str_m_d_y_h_m_s = time.strftime("%m-%d-%Y_%H-%M-%S", t)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(args.logging_dir, f"{str_m_d_y_h_m_s}.log")
            ),
        ]
        if accelerator.is_main_process
        else [],
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    # Load model
    model = MultiBlipDisenBooth(args)
    # with open('/LAVIS/multi_BlipDisenBooth/multi_BLIP_DisenBooth.txt', 'w') as f:
    #         for name, param in model.named_parameters():
    #             f.write(f"{name}\n")
    #             f.write(f"{param}\n")
    
    # if args.load_model is not None:
    #     model.load_state_dict(
    #         torch.load(Path(args.load_model) / "pytorch_model.bin", map_location="cpu")
    #     )
    
    if args.train_text_encoder:
        model.text_encoder.requires_grad(args.train_text_encoder)

    model.to(accelerator.device, dtype = weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
        model.img_adapter.enable_gradient_checkpointing()
        model.proj_layer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            model.text_encoder.gradient_checkpointing_enable()
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW

    unet_params = list([p for p in model.unet.parameters() if p.requires_grad])
    proj_layer_params = list([p for p in model.proj_layer.parameters() if p.requires_grad])
    img_adapter_params = list([p for p in model.img_adapter.parameters() if p.requires_grad])
    if args.train_text_encoder:
        text_encoder_params = list([p for p in model.text_encoder.parameters() if p.requires_grad])
    
    optimizer_params = [
        {"params": unet_params, "lr": args.learning_rate * args.unet_lr_scale},
        {"params": proj_layer_params, "lr": args.learning_rate * args.proj_lr_scale},
        {"params": img_adapter_params, "lr": args.learning_rate * args.img_adapter_lr_scale},
    ]

    if args.train_text_encoder:
        text_encoder_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
        optimizer_params.append({"params": text_encoder_params, "lr": args.learning_rate * args.text_encoder_lr_scale})

    optimizer = optimizer_class(
        optimizer_params,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)