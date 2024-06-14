"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from processers import BlipDiffusionInputImageProcessor, BlipDiffusionTargetImageProcessor, text_proceesser


class SubjectDrivenTextToImageDataset(Dataset):
    def __init__(
        self,
        image_dir,
        subject_text,
        text_prompt,
        text_tokenizer,
        repetition=100000,
    ):
        self.text_tokenizer = text_tokenizer

        self.subject = text_proceesser(subject_text.lower())
        self.image_dir = image_dir

        self.inp_image_transform = BlipDiffusionInputImageProcessor()
        self.tgt_image_transform = BlipDiffusionTargetImageProcessor()


        image_paths = os.listdir(image_dir)
        # if not Path(image_paths).exists():
        #     raise ValueError("Images root doesn't exists.")
        # image paths are jpg png webp
        image_paths = [
            os.path.join(image_dir, imp)
            for imp in image_paths
            if os.path.splitext(imp)[1][1:]
            in ["jpg", "png", "webp", "jpeg", "JPG", "PNG", "WEBP", "JPEG"]
        ]
        # make absolute path
        self.image_paths = [os.path.abspath(imp) for imp in image_paths]
        self.repetition = repetition

    def __len__(self):
        return len(self.image_paths) * self.repetition
    
    @property
    def len_without_repeat(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(image_path).convert("RGB")

        # For fine-tuning, we use the same caption for all images
        # maybe worth trying different captions for different images
        caption = f"a {self.subject}"
        caption = text_proceesser(caption)
        caption_ids = self.text_tokenizer(
            caption[0],
            padding="do_not_pad",
            truncation=True,
            max_length=self.text_tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        inp_image = self.inp_image_transform(image)
        tgt_image = self.tgt_image_transform(image)
        # print(caption_ids)
        return {
            "inp_image": inp_image,
            "tgt_image": tgt_image,
            "subject_text": self.subject,
            "caption_ids": caption_ids,            
        }

def collate_fn(samples):
    samples = [s for s in samples if s is not None]
    # Check if samples is empty after filtering
    if not samples:
        return {}
    collated_dict = {}
    keys = samples[0].keys() # Use the keys of the first sample as a reference
    for k in keys:
        values = [sample[k] for sample in samples]
        # If the value type for the key is torch.Tensor, stack them else return list
        collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
    return collated_dict

