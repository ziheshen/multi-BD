from utils import infer_parse_args
from multi_BD import MultiBlipDisenBooth
import torch
from safetensors.torch import load_file
import logging, time, os, sys
from PIL import Image

def inference(args):
    logging_dir = "/LAVIS/multi_BlipDisenBooth/infer_logs"
    if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)

    t = time.localtime()
    str_m_d_y_h_m_s = time.strftime("%m-%d-%Y_%H-%M-%S", t)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(logging_dir, f"{str_m_d_y_h_m_s}.log")
            ),
        ]
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device")

    # Load model
    model = MultiBlipDisenBooth(args)
    model.to(device)

    logging.info(f"Loading finetuned model from {args.finetuned_ckpt}...")
    # finetuned_ckpt = args.finetuned_ckpt
    # finetuned_ckpt = load_file(finetuned_ckpt, device="cpu")
    # print(finetuned_ckpt.keys)
    model.load_checkpoint(args.finetuned_ckpt)
    # return
    cond_image = Image.open("/LAVIS/data/mit_mascot/mit_mascot1.png")
    cond_subject = ["mit_mascot"]
    tgt_subject = ["mit_mascot"]
    text_prompt = ["on the beach"]
    samples = {
        "cond_images": cond_image,
        "cond_subject": cond_subject,
        "tgt_subject": tgt_subject,
        "prompt": text_prompt,
    }

    num_output = 4

    iter_seed = 8888
    guidance_scale = 7.5
    num_inference_steps = 100
    negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

    for i in range(num_output):
        output = model.generate(
            samples,
            seed=iter_seed + i,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            neg_prompt=negative_prompt,
            height=512,
            width=512,
        )

        img = output[0]

        img.save(f"./result_img/mit_mascot_bsz_2_with_irrel_and_rel/checkpoint_200/img_{i}.png")

    torch.cuda.empty_cache()

if __name__ == "__main__":
    args = infer_parse_args()
    inference(args)