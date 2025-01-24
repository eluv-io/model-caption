from transformers import AutoProcessor, AutoModelForCausalLM
from typing import List, Union
from common_ml.tags import FrameTag
from common_ml.model import FrameModel
from dataclasses import dataclass, asdict
from common_ml.types import Data
from config import config as global_config

import re
import numpy as np
from PIL import Image
from loguru import logger

@dataclass
class RuntimeConfig(Data):
    fps: int
    ipt_rgb: bool
    allow_single_frame: bool

    @staticmethod
    def from_dict(data: dict) -> 'RuntimeConfig':
        return RuntimeConfig(**data)

class CaptionModel(FrameModel):
    def __init__(self, weights: str, config: Union[dict, RuntimeConfig]):
        logger.info("loading caption model")
        self.caption_model = AutoModelForCausalLM.from_pretrained(weights)
        self.caption_processor = AutoProcessor.from_pretrained(weights)

        if isinstance(config, dict):
            config = RuntimeConfig.from_dict(config)
        self.config = config

        self.device = global_config["device"]
        logger.info(self.device)
        self.caption_model = self.caption_model.to(self.device)

        self.patterns = [
            r"(with|and)\s(a|the)\s(words*|letters*|numbers*)",
            r"that\s(have|has)\s(a|the)\s(words*|letters*|numbers*)",
            r"with\s(a|the)\ssign",
            r"that\ssays+"
        ]
        
    def tag(self, img: np.ndarray) -> List[FrameTag]: 
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            if not self.config.ipt_rgb:
                img = img.convert("RGB")
        pixel_values = self.caption_processor(images=img, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.caption_model.generate(pixel_values=pixel_values, max_length=50, num_beams=4).cpu()
        generated_caption = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # TODO need to reconsider if it is a good approach
        for pat in self.patterns:
            search_res = re.search(pat, generated_caption)
            if search_res is not None:
                generated_caption = generated_caption[:search_res.span()[0]] + "."
                break
        return [FrameTag.from_dict({"text": generated_caption, "confidence": 1.0, "box": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}})]
    
    def set_config(self, data: dict) -> None:
        self.config = RuntimeConfig.from_dict(data)

    def get_config(self) -> dict:
        return asdict(self.config)