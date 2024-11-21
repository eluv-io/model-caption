
import argparse
import os
import json
from loguru import logger
from marshmallow import Schema, fields, ValidationError
from typing import List
from dataclasses import asdict

from caption.model import CaptionModel
from config import config

class RuntimeConfig(Schema):
    freq = fields.Int()
    ipt_rgb = fields.Bool()
    allow_single_frame = fields.Bool()

def run(video_paths: List[str], runtime_config: str=None):
    files = video_paths
    if runtime_config is None:
        cfg = config["runtime"]["default"]
    else:
        with open(runtime_config, 'r') as fin:
            cfg = json.load(fin)
    try:
        runtime_config = RuntimeConfig().load(cfg)
    except ValidationError as e:
        logger.error("Received invalid runtime config.")
        raise e
    filedir = os.path.dirname(os.path.abspath(__file__))
    tags_out = os.getenv('TAGS_PATH', os.path.join(filedir, 'tags'))
    if not os.path.exists(tags_out):
        os.makedirs(tags_out)
    model = CaptionModel(runtime_config["ipt_rgb"], config["weights"])
    for fname in files:
        logger.info(f"Tagging video: {fname}")
        ftags, vtags = model.tag_video(fname, runtime_config["allow_single_frame"], runtime_config["freq"])
        with open(os.path.join(tags_out, f"{os.path.basename(fname).split('.')[0]}_tags.json"), 'w') as fout:
            fout.write(json.dumps([asdict(tag) for tag in vtags]))
        with open(os.path.join(tags_out, f"{os.path.basename(fname).split('.')[0]}_frametags.json"), 'w') as fout:
            ftags = {k: [asdict(tag) for tag in v] for k, v in ftags.items()}
            fout.write(json.dumps(ftags))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_paths', nargs='+', type=str)
    parser.add_argument('--config', type=str, required=False)
    args = parser.parse_args()
    run(args.video_paths, args.config)