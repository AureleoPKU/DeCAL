from __future__ import annotations
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from transformers.models.qwen3_vl import Qwen3VLProcessor

import torch

from lerobot.utils.constants import (
    ACTION, 
    OBS_IMAGE, OBS_IMAGES, OBS_TACTILE_IMAGES, OBS_STATE, OBS_STR, 
)
from lerobot.transforms.core import DataTransformFn, DataDict


DECAL_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QWEN_CACHE_DIR = DECAL_ROOT / "ckpt" / "qwen"


@DataTransformFn.register_subclass("decal_processor")
@dataclass
class DeCALProcessorTransformFn(DataTransformFn):
    pretrained_model_name_or_path: str = 'Qwen/Qwen3-VL-2B-Instruct'
    max_length: int = 48
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True

    spatial_merge_size: int = 2

    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151655

    process: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        processor_path = os.environ.get(
            "DECAL_QWEN_PROCESSOR_PATH", self.pretrained_model_name_or_path
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            processor_path,
            cache_dir=os.environ.get("DECAL_QWEN_CACHE_DIR", str(DEFAULT_QWEN_CACHE_DIR)),
        )
        self.vision_start_token_id = self.processor.vision_start_token_id
        self.vision_end_token_id = self.processor.vision_end_token_id
        self.image_token_id = self.processor.image_token_id

    def __call__(self, data: DataDict) -> DataDict: 
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        for i in range(3):
            k = f"{OBS_IMAGES}.image{i}"
            if data[f"{k}_mask"]:
                img_inputs = self.processor.image_processor(
                    data[k][1],  # we only feed images at current time to vlm
                    do_rescale=False, 
                )
                pixel_values.append(img_inputs.pixel_values)
                image_grid_thw.append(img_inputs.image_grid_thw)
                num_img_token = torch.prod(image_grid_thw[-1]) // self.spatial_merge_size ** 2
                input_ids += [self.vision_start_token_id] + [self.image_token_id] * num_img_token + [self.vision_end_token_id]
                attention_mask += [1] * (num_img_token + 2)
                # attention_mask += [0] + [1] * num_img_token + [0]
            else:
                pixel_values.append(img_inputs.pixel_values)
                image_grid_thw.append(img_inputs.image_grid_thw)
                input_ids += [self.vision_start_token_id] + [self.image_token_id] * num_img_token + [self.vision_end_token_id]
                attention_mask += [0] * (num_img_token + 2)
        
        data[f"{OBS_STR}.pixel_values"] = torch.cat(pixel_values)
        data[f"{OBS_STR}.image_grid_thw"] = torch.cat(image_grid_thw)

        lang_inputs = self.processor.tokenizer(
            data[self.task_key], 
            max_length=self.max_length, 
            padding_side=self.padding_side, 
            padding=self.padding, 
            truncation=self.truncation, 
        )

        input_ids += lang_inputs.input_ids
        attention_mask += lang_inputs.attention_mask
        data[f"{OBS_STR}.input_ids"] = torch.tensor(input_ids)
        data[f"{OBS_STR}.attention_mask"] = torch.tensor(attention_mask)

        return data
    

@DataTransformFn.register_subclass("unify_decal_inputs")
@dataclass
class UnifyDeCALInputsTransformFn(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict: 
        data = {
            OBS_STATE: data[OBS_STATE], 
            ACTION: data[ACTION], 
            f"{OBS_STR}.tactile": data[f"{OBS_STR}.tactile"],
            f"{OBS_IMAGES}.image0": data[f"{OBS_IMAGES}.image0"], 
            f"{OBS_IMAGES}.image1": data[f"{OBS_IMAGES}.image1"], 
            f"{OBS_IMAGES}.image2": data[f"{OBS_IMAGES}.image2"], 
            f"{OBS_IMAGES}.image0_mask": data[f"{OBS_IMAGES}.image0_mask"], 
            f"{OBS_IMAGES}.image1_mask": data[f"{OBS_IMAGES}.image1_mask"], 
            f"{OBS_IMAGES}.image2_mask": data[f"{OBS_IMAGES}.image2_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_thumb_deform": data[f"{OBS_TACTILE_IMAGES}.left_thumb_deform"], 
            f"{OBS_TACTILE_IMAGES}.left_index_deform": data[f"{OBS_TACTILE_IMAGES}.left_index_deform"], 
            f"{OBS_TACTILE_IMAGES}.left_middle_deform": data[f"{OBS_TACTILE_IMAGES}.left_middle_deform"], 
            f"{OBS_TACTILE_IMAGES}.left_ring_deform": data[f"{OBS_TACTILE_IMAGES}.left_ring_deform"], 
            f"{OBS_TACTILE_IMAGES}.left_pinky_deform": data[f"{OBS_TACTILE_IMAGES}.left_pinky_deform"], 
            f"{OBS_TACTILE_IMAGES}.right_thumb_deform": data[f"{OBS_TACTILE_IMAGES}.right_thumb_deform"], 
            f"{OBS_TACTILE_IMAGES}.right_index_deform": data[f"{OBS_TACTILE_IMAGES}.right_index_deform"], 
            f"{OBS_TACTILE_IMAGES}.right_middle_deform": data[f"{OBS_TACTILE_IMAGES}.right_middle_deform"], 
            f"{OBS_TACTILE_IMAGES}.right_ring_deform": data[f"{OBS_TACTILE_IMAGES}.right_ring_deform"], 
            f"{OBS_TACTILE_IMAGES}.right_pinky_deform": data[f"{OBS_TACTILE_IMAGES}.right_pinky_deform"], 
            f"{OBS_TACTILE_IMAGES}.left_thumb_deform_mask": data[f"{OBS_TACTILE_IMAGES}.left_thumb_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_index_deform_mask": data[f"{OBS_TACTILE_IMAGES}.left_index_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_middle_deform_mask": data[f"{OBS_TACTILE_IMAGES}.left_middle_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_ring_deform_mask": data[f"{OBS_TACTILE_IMAGES}.left_ring_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_pinky_deform_mask": data[f"{OBS_TACTILE_IMAGES}.left_pinky_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_thumb_deform_mask": data[f"{OBS_TACTILE_IMAGES}.right_thumb_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_index_deform_mask": data[f"{OBS_TACTILE_IMAGES}.right_index_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_middle_deform_mask": data[f"{OBS_TACTILE_IMAGES}.right_middle_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_ring_deform_mask": data[f"{OBS_TACTILE_IMAGES}.right_ring_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_pinky_deform_mask": data[f"{OBS_TACTILE_IMAGES}.right_pinky_deform_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_thumb_raw": data[f"{OBS_TACTILE_IMAGES}.left_thumb_raw"], 
            f"{OBS_TACTILE_IMAGES}.left_index_raw": data[f"{OBS_TACTILE_IMAGES}.left_index_raw"], 
            f"{OBS_TACTILE_IMAGES}.left_middle_raw": data[f"{OBS_TACTILE_IMAGES}.left_middle_raw"], 
            f"{OBS_TACTILE_IMAGES}.left_ring_raw": data[f"{OBS_TACTILE_IMAGES}.left_ring_raw"], 
            f"{OBS_TACTILE_IMAGES}.left_pinky_raw": data[f"{OBS_TACTILE_IMAGES}.left_pinky_raw"], 
            f"{OBS_TACTILE_IMAGES}.right_thumb_raw": data[f"{OBS_TACTILE_IMAGES}.right_thumb_raw"], 
            f"{OBS_TACTILE_IMAGES}.right_index_raw": data[f"{OBS_TACTILE_IMAGES}.right_index_raw"], 
            f"{OBS_TACTILE_IMAGES}.right_middle_raw": data[f"{OBS_TACTILE_IMAGES}.right_middle_raw"], 
            f"{OBS_TACTILE_IMAGES}.right_ring_raw": data[f"{OBS_TACTILE_IMAGES}.right_ring_raw"], 
            f"{OBS_TACTILE_IMAGES}.right_pinky_raw": data[f"{OBS_TACTILE_IMAGES}.right_pinky_raw"], 
            f"{OBS_TACTILE_IMAGES}.left_thumb_raw_mask": data[f"{OBS_TACTILE_IMAGES}.left_thumb_raw_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_index_raw_mask": data[f"{OBS_TACTILE_IMAGES}.left_index_raw_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_middle_raw_mask": data[f"{OBS_TACTILE_IMAGES}.left_middle_raw_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_ring_raw_mask": data[f"{OBS_TACTILE_IMAGES}.left_ring_raw_mask"], 
            f"{OBS_TACTILE_IMAGES}.left_pinky_raw_mask": data[f"{OBS_TACTILE_IMAGES}.left_pinky_raw_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_thumb_raw_mask": data[f"{OBS_TACTILE_IMAGES}.right_thumb_raw_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_index_raw_mask": data[f"{OBS_TACTILE_IMAGES}.right_index_raw_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_middle_raw_mask": data[f"{OBS_TACTILE_IMAGES}.right_middle_raw_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_ring_raw_mask": data[f"{OBS_TACTILE_IMAGES}.right_ring_raw_mask"], 
            f"{OBS_TACTILE_IMAGES}.right_pinky_raw_mask": data[f"{OBS_TACTILE_IMAGES}.right_pinky_raw_mask"], 
            f"{OBS_STR}.pixel_values": data[f"{OBS_STR}.pixel_values"], 
            f"{OBS_STR}.image_grid_thw": data[f"{OBS_STR}.image_grid_thw"], 
            # "task": data["task"], 
            f"{OBS_STR}.input_ids": data[f"{OBS_STR}.input_ids"], 
            f"{OBS_STR}.attention_mask": data[f"{OBS_STR}.attention_mask"], 
        }
        return data


if __name__ == "__main__":
    sample = {
        f"{OBS_IMAGES}.image0": torch.rand((224, 224, 3)), 
        f"{OBS_IMAGES}.image1": torch.rand((224, 224, 3)), 
        f"{OBS_IMAGES}.image2": torch.rand((224, 224, 3)), 
        "task": "This is a test sample.", 
    }
    processor = DeCALProcessorTransformFn()
    output = processor(sample)
    print(output)
