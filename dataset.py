"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from processors import input_image_precesser, target_image_processer, text_proceesser


class SubjectDrivenTextToImageDataset(Dataset):
    def __init__(
        self,
        image_dir,
        subject_text,
        text_prompt,
        repetition=100000,
    ):
        self.subject = text_proceesser(subject_text.lower())
        self.image_dir = image_dir

        self.inp_image_transform = input_image_precesser()
        self.tgt_image_transform = target_image_processer()


        image_paths = os.listdir(image_dir)
        if not Path(image_paths).exists():
            raise ValueError("Images root doesn't exists.")
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

    def collater(self, samples):
        return default_collate(samples)

    def __getitem__(self, index):
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(image_path).convert("RGB")

        # For fine-tuning, we use the same caption for all images
        # maybe worth trying different captions for different images
        caption = f"a {self.subject}"
        caption = text_proceesser(caption)

        inp_image = self.inp_image_transform(image)
        tgt_image = self.tgt_image_transform(image)

        return {
            "inp_image": inp_image,
            "tgt_image": tgt_image,
            "caption": caption,
            "subject_text": self.subject,
        }
    