import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig
import open_clip
from disen_net import Image_adapter, cal_cos
import torch.nn as nn
import logging
import os

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
# from diffusers.optimization import get_scheduler
# from diffusers.training_utils import unet_lora_state_dict
# from diffusers.utils import check_min_version, is_wandb_available
# from diffusers.utils.import_utils import is_xformers_available

from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_diffusion_models.modeling_ctx_clip import CtxCLIPTextModel
from lavis.common.utils import download_and_untar, is_url
from transformers.activations import QuickGELUActivation as QuickGELU

class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        # self.dense1 = nn.Linear(in_dim, hidden_dim)
        # self.act_fn = QuickGELU()
        # self.dense2 = nn.Linear(hidden_dim, out_dim)
        # self.dropout = nn.Dropout(drop_p)

        # self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)
        self.dense_in = nn.Linear(in_dim, out_dim)
        self.dense1 = nn.Linear(out_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)
        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x):
        # x_in = x

        # x = self.LayerNorm(x)
        # x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in
        x = self.dense_in(x)  # 先轉換維度到1024
        x_in = x
        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x
    
class MultiBlipDisenBooth(nn.Module):
    def __init__(
        self, args,
        qformer_num_query_token=16,
        qformer_cross_attention_freq=1,
        qformer_pretrained_path=None,
        qformer_train=False,
        pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base",
        train_text_encoder=False,
        vae_half_precision=True,
        proj_train=True,
        img_adapter_train=True,
        img_encoder_train=False,
        subject_text="",
        text_prompt="",
        id_rel_weight = 1.0,
        id_irrel_weight = 1.0,
    ):
        super().__init__()
        
        ### Identity-relevent Branch ###
        # BLIP2 QFormer
        self.num_query_token = args.qformer_num_query_token
        self.subject_text = args.validation_prompt
        

        self.blip = Blip2Qformer(
            vit_model="clip_L",
            num_query_token=args.qformer_num_query_token,
            cross_attention_freq=args.qformer_cross_attention_freq,
        )
        # if args.qformer_pretrained_path is not None:
        #     state_dict = torch.load(args.qformer_pretrained_path, map_location="cpu")[
        #         "model"
        #     ]
        #     # qformer keys: Qformer.bert.encoder.layer.1.attention.self.key.weight
        #     # ckpt keys: text_model.bert.encoder.layer.1.attention.self.key.weight
        #     for k in list(state_dict.keys()):
        #         if "text_model" in k:
        #             state_dict[k.replace("text_model", "Qformer")] = state_dict.pop(k)

        #     msg = self.blip.load_state_dict(state_dict, strict=False)
        #     assert all(["visual" in k for k in msg.missing_keys])
        #     assert len(msg.unexpected_keys) == 0
        
        self.qformer_train = args.qformer_train

        # Projection Layer
        self.proj_layer = ProjLayer(
            in_dim=768, out_dim=768, hidden_dim=3072, drop_p=0.1, eps=1e-12
        )
        self.proj_train = args.proj_train

        # Text Encoder
        self.text_prompt = args.text_prompt

        self.tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CtxCLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder"
        )

        # self.id_rel_weight = args.id_rel_weight

        ### Identity-irrelevent Branch ###
        # Image Encoder
        self.img_encoder, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
        self.clip_trans = transforms.Resize( (224, 224), interpolation=transforms.InterpolationMode.BILINEAR )
        
        self.img_encoder_train = args.img_encoder_train

        # Adapter
        self.img_adapter = Image_adapter()

        self.img_adapter_train = args.img_adapter_train

        # self.id_irrel_weight = args.id_irrel_weight

        ### Stable Diffusion ###
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        )
        if args.vae_half_precision:
            self.vae.half()

        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet"
        )
        # self.unet.enable_xformers_memory_efficient_attention()

        self.noise_scheduler = DDPMScheduler.from_config(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.train_text_encoder = args.train_text_encoder

        ### Other Settings ###
        self.freeze_modules()
        self.ctx_embeddings_cache = nn.Parameter(
            torch.zeros(1, self.num_query_token, 768), requires_grad=False
        )
        self._use_embeddings_cache = False
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # inference-related
        self._CTX_BEGIN_POS = 2

        ### Load parameters from pretrained BLIP-Diffusion ###
        self.pretrained_BLIPdiffusion_name_or_path = args.pretrained_BLIPdiffusion_name_or_path
        self.load_pretrained_parameters()

    def freeze_modules(self):
        to_freeze = [self.vae]
        if not self.train_text_encoder:
            to_freeze.append(self.text_encoder)

        if not self.qformer_train:
            to_freeze.append(self.blip)

        if not self.proj_train:
            to_freeze.append(self.proj_layer)
        
        if not self.img_encoder_train:
            to_freeze.append(self.img_encoder)
        
        if not self.img_adapter_train:
            to_freeze.append(self.img_adapter)

        for module in to_freeze:
            module.eval()
            module.train = self.disabled_train
            module.requires_grad_(False)

    def disabled_train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    def load_pretrained_parameters(self):
        if is_url(self.pretrained_BLIPdiffusion_name_or_path):
            checkpoint_dir_or_url = download_and_untar(self.pretrained_BLIPdiffusion_name_or_path)
        
        logging.info(f"Loading pretrained model from {checkpoint_dir_or_url}")
        
        def load_state_dict(module, filename):
            try:
                state_dict = torch.load(
                    os.path.join(checkpoint_dir_or_url, filename), map_location="cpu"
                )
                msg = module.load_state_dict(state_dict, strict=False)
            except FileNotFoundError:
                logging.info("File not found, skip loading: {}".format(filename))
        
        # state_dict = torch.load(self.pretrained_BLIPdiffusion_name_or_path, map_location="cpu")["model"]

        # Load parameters for BLIP2 QFormer
        logging.info("Loading pretrained BLIP2 Qformer weights.")
        load_state_dict(self.blip, "blip_model/blip_weight.pt")
    

        # Load parameters for the projection layer
        logging.info("Loading pretrained projection lyer weights.")
        load_state_dict(self.proj_layer, "proj_layer/proj_weight.pt")

        # Load parameters for the text encoder
        logging.info("Loading pretrained text encoder weights.")
        load_state_dict(self.text_encoder, "text_encoder/pytorch_model.bin")

         # Load parameters for the vae
        logging.info("Loading pretrained vae weights.")
        load_state_dict(self.vae, "vae/diffusion_pytorch_model.bin")

         # Load parameters for the unet
        logging.info("Loading pretrained unet weights.")
        load_state_dict(self.unet, "unet/diffusion_pytorch_model.bin")

             

    def forward(self, samples):
        latents = self.vae.encode(samples["tgt_image"].half()).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        ### Identity-irrelevent Branch ###
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the BLIP2 Qformer image-text embedding
        ctx_embeddings = self.forward_ctx_embeddings(
            input_image=samples["inp_image"], text_input=samples["subject_text"]
        )

        # Get the text embedding for conditioning
        input_ids = self.tokenizer(
            samples["caption"],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)

        encoder_hidden_states = self.text_encoder(
            input_ids=input_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=[self._CTX_BEGIN_POS] * input_ids.shape[0],
        )[0]

        ### Identity-irrelevent Branch ###
        # Get the image embedding
        img_embeddings = self.img_encoder.encode_image( self.clip_trans(samples["inp_image"]) ).unsqueeze(1)
        img_state = self.img_adapter( img_embeddings )

        # Predict the noise residual
        id_rel_pred = self.unet(
            noisy_latents.float(), timesteps, encoder_hidden_states + img_state
        ).sample
        id_irrel_pred = self.unet(
            noisy_latents.float(), timesteps, img_state
        ).sample

        # if model predicts variance, throw away the prediction. we will only train on the
        # simplified training objective. This means that all schedulers using the fine tuned
        # model must be configured to use one of the fixed variance variance types.
        if id_rel_pred.shape[1] == 6:
            id_rel_pred, _ = torch.chunk(id_rel_pred, 2, dim=1)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        loss = F.mse_loss(id_rel_pred.float(), target.float(), reduction="mean")
        loss_aux1 = F.mse_loss(id_irrel_pred.float(), target.float(), reduction="mean")
        loss_aux2 = cal_cos(encoder_hidden_states, img_state, self.cos)
        loss = loss + 0.01*loss_aux1 + 0.001*loss_aux2

        return { "loss": loss }
        
    def forward_ctx_embeddings(self, input_image, text_input, ratio=None):
        def compute_ctx_embeddings(input_image, text_input):
            # blip_embeddings = self.blip(image=input_image, text=text_input)
            blip_embeddings = self.blip.extract_features(
                {"image": input_image, "text_input": text_input}, mode="multimodal"
            ).multimodal_embeds
            ctx_embeddings = self.proj_layer(blip_embeddings)
            return ctx_embeddings

        if isinstance(text_input, str):
            text_input = [text_input]

        if self._use_embeddings_cache:
            # expand to batch size
            ctx_embeddings = self.ctx_embeddings_cache.expand(len(text_input), -1, -1)
        else:
            if isinstance(text_input[0], str):
                text_input, input_image = [text_input], [input_image]

            all_ctx_embeddings = []

            for inp_image, inp_text in zip(input_image, text_input):
                ctx_embeddings = compute_ctx_embeddings(inp_image, inp_text)
                all_ctx_embeddings.append(ctx_embeddings)

            if ratio is not None:
                assert len(ratio) == len(all_ctx_embeddings)
                assert sum(ratio) == 1
            else:
                ratio = [1 / len(all_ctx_embeddings)] * len(all_ctx_embeddings)

            ctx_embeddings = torch.zeros_like(all_ctx_embeddings[0])
            #print('===============================\n\nall_ctx_embeddings:', all_ctx_embeddings[0].size())
            for ratio, ctx_embeddings_ in zip(ratio, all_ctx_embeddings):
                ctx_embeddings += ratio * ctx_embeddings_
            #test embedding shape
            
            # print('ctx_embeddings:', ctx_embeddings.size())
            # print('\n\n================================')
        
        return ctx_embeddings