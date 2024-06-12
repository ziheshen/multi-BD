import re

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def text_proceesser( caption, max_words=50 ):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[: max_words])

    return caption


class BlipDiffusionInputImageProcessor():
    def __init__(
        self,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std = (0.26862954, 0.26130258, 0.27577711),
        image_size=224
    ):

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )

    def __call__(self, item):
        return self.transform(item)

class BlipDiffusionTargetImageProcessor():
    def __init__(
        self,
        image_size=512,
    ):

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __call__(self, item):
        return self.transform(item)