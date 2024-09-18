from transformers import AutoProcessor, AutoModelForCausalLM
from typing import List
from config import config
from common_ml.tags import FrameTag
from common_ml.model import FrameModel

import re
import numpy as np
from PIL import Image
from loguru import logger
from config import config

class CaptionModel(FrameModel):
    def __init__(self, ipt_rgb: bool, weights: str):
        logger.info("loading caption model")
        self.caption_model = AutoModelForCausalLM.from_pretrained(weights)
        self.caption_processor = AutoProcessor.from_pretrained(weights)

        self.device = config["device"]
        self.ipt_rgb = ipt_rgb
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
            img = Image.fromarray(img[:, :, ::-1])
            if not self.ipt_rgb:
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
        return [FrameTag(text=generated_caption, confidence=1.0, box=(0.05, 0.05, 0.95, 0.95))]