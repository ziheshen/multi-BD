from utils import parse_args
from multi_BD import MultiBlipDisenBooth
import torch

def inference(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Load model
    model = MultiBlipDisenBooth(args)
    model.to(device)

    finetuned_ckpt = args.finetuned_ckpt
    finetuned_ckpt = torch.load(finetuned_ckpt, map=device)
    model.load_state_dict(finetuned_ckpt['model_state_dict'])

if __name__ == "__main__":
    args = parse_args()
    inference(args)