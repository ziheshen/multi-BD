
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
from dataset import SubjectDrivenTextToImageDataset, collate_fn
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
            logging.info("Using xformers.")
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

    logging.info("Building datasets...")
    train_dataset = SubjectDrivenTextToImageDataset(
        image_dir=args.instance_data_dir,
        subject_text=args.subject_text,
        text_prompt=args.text_prompt,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.train_batch_size,
        shuffle = True,
        collate_fn = lambda samples: collate_fn(samples),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers(
            project_name="BLIP-DisenBooth",
            config=tracker_config
        )
    
    # Train!
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         logger.info(f"Trainable parameter: {name} with shape {param.shape}")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("\n***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save

    initial_global_step = 0
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            progress_bar.set_description("Global step: {}".format(global_step))

            with accelerator.accumulate(model), torch.backends.cuda.sdp_kernel(
                enable_flash=not args.disable_flashattention
            ):
                # if step == 0:
                #     model.before_training(model=model,
                #                           dataset=batch
                #                           )
                return_dict = model(batch)
                loss = return_dict['loss']

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        unet_params + proj_layer_params + img_adapter_params +text_encoder_params
                        if args.train_text_encoder
                        else unet_params + proj_layer_params + img_adapter_params
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    
    # Save the every layers
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     model = accelerator.unwrap_model(model)

    #     pipeline = model.to_pipeline()
    #     pipeline.save_pretrained(args.output_dir)
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    train(args)