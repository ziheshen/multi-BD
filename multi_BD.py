import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
# from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig
import open_clip
from disen_net import Image_adapter, cal_cos
import torch.nn as nn
import logging
import os
import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnAddedKVProcessor,
)

from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip_diffusion_models.modeling_ctx_clip import CtxCLIPTextModel
from lavis.common.utils import download_and_untar, is_url
from lavis.models.blip_diffusion_models.ptp_utils import (
    LocalBlend,
    P2PCrossAttnProcessor,
    AttentionRefine,
)
from lavis.models.blip_diffusion_models.utils import numpy_to_pil
from lavis.common.dist_utils import download_cached_file
from transformers.activations import QuickGELUActivation as QuickGELU

from safetensors.torch import load_file

class ProjLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        super().__init__()

        # Dense1 -> Act -> Dense2 -> Drop -> Res -> Norm
        self.dense1 = nn.Linear(in_dim, hidden_dim)
        self.act_fn = QuickGELU()
        self.dense2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(drop_p)

        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)

    def forward(self, x):
        x_in = x

        x = self.LayerNorm(x)
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x
    
class MultiBlipDisenBooth(nn.Module):
    def __init__(
        self,
        args,
        subject_text="",
        text_prompt="",
        qformer_num_query_token=16,
        qformer_cross_attention_freq=1,
        qformer_pretrained_path=None,
        qformer_train=False,
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        pretrained_BLIPdiffusion_name_or_path="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP-Diffusion/blip-diffusion.tar.gz",
        train_text_encoder=False,
        vae_half_precision=True,
        proj_train=False,
        img_adapter_train=True,
        img_encoder_train=False,
        id_rel_weight = 1.0,
        id_irrel_weight = 1.0,
        use_irrel_branch=True
    ):
        super().__init__()
        
        ### Identity-relevent Branch ###
        # BLIP2 QFormer
        self.num_query_token = args.qformer_num_query_token
        # self.subject_text = args.validation_prompt
        

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
        # self.text_prompt = args.text_prompt

        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     pretrained_model_name_or_path, subfolder="tokenizer"
        # )
        self.text_encoder = CtxCLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )

        # self.id_rel_weight = args.id_rel_weight

        ### Identity-irrelevent Branch ###
        # Image Encoder
        self.img_encoder, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
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
        # if args.vae_half_precision:
        #     self.vae.half()

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

    def device(self):
        return list(self.parameters())[0].device

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

    @property
    def pndm_scheduler(self):
        if not hasattr(self, "_pndm_scheduler"):
            self._pndm_scheduler = PNDMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                set_alpha_to_one=False,
                skip_prk_steps=True,
            )
        return self._pndm_scheduler

    @property
    def ddim_scheduler(self):
        if not hasattr(self, "_ddim_scheduler"):
            self._ddim_scheduler = DDIMScheduler.from_config(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            )
        return self._ddim_scheduler

    def before_training(self, dataset, **kwargs):
        # print(dataset)
        # assert len(dataset) == 1, "Only support single dataset for now."

        # key = list(dataset.keys())[0]
        # dataset = dataset[key]["train"]

        # collect all examples
        # [FIXME] this is not memory efficient. may OOM if the dataset is large.
        examples = [dataset[i] for i in range(dataset.len_without_repeat)]
        # print(examples)
        input_images = (
            torch.stack([example["inp_image"] for example in examples])
            .to(memory_format=torch.contiguous_format)
            .float()
        ).to(self.device())
        subject_text = [dataset.subject for _ in range(input_images.shape[0])]
        
        # calculate ctx embeddings and cache them
        ctx_embeddings = self.forward_ctx_embeddings(
            input_image=input_images, text_input=subject_text
        )
        # take mean of all ctx embeddings
        ctx_embeddings = ctx_embeddings.mean(dim=0, keepdim=True)
        self.ctx_embeddings_cache = nn.Parameter(ctx_embeddings, requires_grad=True)
        self._use_embeddings_cache = True

        # free up CUDA memory
        self.blip.to("cpu")
        self.proj_layer.to("cpu")

        torch.cuda.empty_cache()

    def forward(self, samples):
        # latents = self.vae.encode(samples["tgt_image"].half()).latent_dist.sample()
        latents = self.vae.encode(samples["tgt_image"]).latent_dist.sample()
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

        ### Identity-relevent Branch ###
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the BLIP2 Qformer image-text embedding
        ctx_embeddings = self.forward_ctx_embeddings(
            input_image=samples["inp_image"], text_input=samples["subject_text"]
        )

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(
            input_ids=samples["caption_ids"],
            # input_ids=input_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=[self._CTX_BEGIN_POS] * samples["caption_ids"].shape[0],
        )[0]

        ### Identity-irrelevent Branch ###
        # Get the image embedding
        img_embeddings = self.img_encoder.encode_image( self.clip_trans(samples["inp_image"]) ).unsqueeze(1)
        # img_embeddings.mean(dim=0, keepdim=True)
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
        loss = loss + 1*loss_aux1 + 1*loss_aux2
        # loss = loss + 0.01*loss_aux1 + 0.001*loss_aux2
        # loss = loss + 0.01*loss_aux2

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

    @torch.no_grad()
    def generate(
            self,
            samples,
            latents=None,
            guidance_scale=7.5,
            height=512,
            width=512,
            seed=42,
            num_inference_steps=50,
            neg_prompt="",
            controller=None,
            prompt_strength=1.0,
            prompt_reps=20,
            use_ddim=False,
            text_weight = 1.0,
            image_weight = 0.0,
    ):
        if controller is not None:
            self._register_attention_refine(controller)
        
        ref_image = samples["ref_images"]
        cond_image = samples["cond_images"]  # reference image
        cond_subject = samples["cond_subject"]  # source subject category
        tgt_subject = samples["tgt_subject"]  # target subject category
        prompt = samples["prompt"]
        cldm_cond_image = samples.get("cldm_cond_image", None)  # conditional image

        ref_img = self.preprocess(ref_image).unsqueeze(0).to("cuda")
        img_feature = self.img_encoder.encode_image( ref_img ).unsqueeze(1) 
        if self.img_adapter is not None:
            img_feature = self.img_adapter(img_feature)
        img_feature = torch.cat([torch.zeros_like(img_feature),img_feature])

        # cond_image = None
        prompt = self._build_prompt(
            prompts=prompt,
            tgt_subjects=tgt_subject,
            prompt_strength=prompt_strength,
            prompt_reps=prompt_reps,
        )

        text_embeddings = self._forward_prompt_embeddings(
            cond_image, cond_subject, prompt
        )        

        # 3. unconditional embedding
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            max_length = self.text_encoder.text_model.config.max_position_embeddings

            uncond_input = self.tokenizer(
                [neg_prompt],
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                input_ids=uncond_input.input_ids.to(self.device()),
                ctx_embeddings=None,
            )[0]

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([text_embeddings, ])
        
        prompt_embeds = text_weight*text_embeddings + image_weight*img_feature

        if seed is not None:
            generator = torch.Generator(device=self.device())
            generator = generator.manual_seed(seed)

        latents = self._init_latent(latents, height, width, generator, batch_size=1)

        scheduler = self.pndm_scheduler if not use_ddim else self.ddim_scheduler

        # set timesteps
        extra_set_kwargs = {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        iterator = tqdm.tqdm(scheduler.timesteps)

        for i, t in enumerate(iterator):
            latents = self._denoise_latent_step(
                latents=latents,
                t=t,
                text_embeddings=prompt_embeds,
                cond_image=cldm_cond_image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                use_inversion=use_ddim,
            )

        image = self._latent_to_image(latents)

        return image

    def _denoise_latent_step(
        self,
        latents,
        t,
        text_embeddings,
        guidance_scale,
        height,
        width,
        cond_image=None,
        use_inversion=False,
    ):
        if use_inversion:
            noise_placeholder = []

        # expand the latents if we are doing classifier free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )

        # predict the noise residual
        noise_pred = self._predict_noise(
            t=t,
            latent_model_input=latent_model_input,
            text_embeddings=text_embeddings,
            width=width,
            height=height,
            cond_image=cond_image,
        )

        if use_inversion:
            noise_placeholder.append(noise_pred[2].unsqueeze(0))

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        if use_inversion:
            noise_placeholder.append(noise_pred[-1].unsqueeze(0))
            noise_pred = torch.cat(noise_placeholder)

        # compute the previous noisy sample x_t -> x_t-1
        scheduler = self.ddim_scheduler if use_inversion else self.pndm_scheduler

        latents = scheduler.step(
            noise_pred,
            t,
            latents,
        )["prev_sample"]

        return latents

    def _register_attention_refine(
        self,
        src_subject,
        prompts,
        num_inference_steps,
        cross_replace_steps=0.8,
        self_replace_steps=0.4,
        threshold=0.3,
    ):
        device, tokenizer = self.device(), self.tokenizer

        lb = LocalBlend(
            prompts=prompts,
            words=(src_subject,),
            device=device,
            tokenizer=tokenizer,
            threshold=threshold,
        )

        controller = AttentionRefine(
            prompts,
            num_inference_steps,
            cross_replace_steps=cross_replace_steps,
            self_replace_steps=self_replace_steps,
            tokenizer=tokenizer,
            device=device,
            local_blend=lb,
        )

        self._register_attention_control(controller)

        return controller
    
    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        rv = []
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # a trick to amplify the prompt
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        return rv
    
    def _tokenize_text(self, text_input, with_query=True):
        max_len = self.text_encoder.text_model.config.max_position_embeddings
        if with_query:
            max_len -= self.num_query_token
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
        )
        tokenized_text = self.tokenizer(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        return tokenized_text
    
    def _forward_prompt_embeddings(self, input_image, src_subject, prompt):
        # 1. extract BLIP query features and proj to text space -> (bs, 32, 768)
        query_embeds = self.forward_ctx_embeddings(input_image, src_subject).to(self.device())

        # 2. embeddings for prompt, with query_embeds as context
        tokenized_prompt = self._tokenize_text(prompt).to(self.device())
        text_embeddings = self.text_encoder(
            input_ids=tokenized_prompt.input_ids,
            ctx_embeddings=query_embeds,
            ctx_begin_pos=[self._CTX_BEGIN_POS],
        )[0]

        return text_embeddings
    
    def _init_latent(self, latent, height, width, generator, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.unet.in_channels, height // 8, width // 8),
                generator=generator,
                device=generator.device,
            )
        latent = latent.expand(
            batch_size,
            self.unet.in_channels,
            height // 8,
            width // 8,
        )
        return latent.to(self.device())
    
    def _predict_noise(
        self,
        t,
        latent_model_input,
        text_embeddings,
        width=512,
        height=512,
        cond_image=None,
    ):
        # if hasattr(self, "controlnet"):
        #     cond_image = prepare_cond_image(
        #         cond_image, width, height, batch_size=1, device=self.device
        #     )

        #     down_block_res_samples, mid_block_res_sample = self.controlnet(
        #         latent_model_input,
        #         t,
        #         encoder_hidden_states=text_embeddings,
        #         controlnet_cond=cond_image,
        #         # conditioning_scale=controlnet_condition_scale,
        #         return_dict=False,
        #     )
        # else:
        #     down_block_res_samples, mid_block_res_sample = None, None
        
        down_block_res_samples, mid_block_res_sample = None, None

        noise_pred = self.unet(
            latent_model_input,
            timestep=t,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )["sample"]

        return noise_pred
    
    def _latent_to_image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        image = numpy_to_pil(image)

        return image
    
    def load_checkpoint(self, url_or_filename):
        """
        Used to load finetuned models.
        """
        # print("self.ctx_embeddings_cache: ",self.ctx_embeddings_cache)
        self._load_checkpoint(url_or_filename)

        # print("loading fine-tuned model from {}".format(url_or_filename))
        self._use_embeddings_cache = True
        # print("self.ctx_embeddings_cache: ",self.ctx_embeddings_cache)


    def _load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = load_file(url_or_filename, device="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg