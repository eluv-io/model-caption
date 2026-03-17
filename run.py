import argparse
from typing import List
from loguru import logger
import setproctitle
import re
import json
import numpy as np
from dataclasses import dataclass
from dacite import from_dict
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

from common_ml.tagging.models.frame_based import FrameModel
from common_ml.tagging.models.tag_types import FrameTag

from config import config

class CaptionModel(FrameModel):
    def __init__(self, weights: str):
        logger.info("loading caption model")
        self.caption_model = AutoModelForCausalLM.from_pretrained(weights)
        self.caption_processor = AutoProcessor.from_pretrained(weights)
        self.device = 'cuda'
        self.caption_model = self.caption_model.to(self.device)

        self.patterns = [
            r"(with|and)\s(a|the)\s(words*|letters*|numbers*)",
            r"that\s(have|has)\s(a|the)\s(words*|letters*|numbers*)",
            r"with\s(a|the)\ssign",
            r"that\ssays+"
        ]
        
    def tag(self, img: np.ndarray) -> List[FrameTag]: 
        img = Image.fromarray(img)
        pixel_values = self.caption_processor(images=img, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.caption_model.generate(pixel_values=pixel_values, max_length=50, num_beams=4).cpu()
        generated_caption = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        for pat in self.patterns:
            search_res = re.search(pat, generated_caption)
            if search_res is not None:
                generated_caption = generated_caption[:search_res.span()[0]] + "."
                break

        return [FrameTag(tag=generated_caption, box={"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95})]
    
@dataclass
class RuntimeConfig:
    fps: float = 1.0
    continue_on_error: bool = False

def _parse_config_string(config_str: str) -> RuntimeConfig:
    try:
        config_dict = json.loads(config_str)
        return from_dict(RuntimeConfig, config_dict)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse config string: {e}")
        raise

if __name__ == '__main__':
    setproctitle.setproctitle("model-caption")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', required=True, type=str, help='Output path (.jsonl)')
    parser.add_argument('--params', type=str, required=False, help='Runtime parameters as JSON')
    
    args = parser.parse_args()
    
    params = _parse_config_string(args.params) if args.params else RuntimeConfig()
    model = CaptionModel(weights=config["model_weights"])
