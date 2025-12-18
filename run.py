import argparse
import os
import json
from typing import List, Optional
import setproctitle

from common_ml.utils import nested_update
from common_ml.model import default_tag, run_live_mode
from caption.model import CaptionModel
from config import config

def get_runtime_config(runtime_config: Optional[str] = None):
    """Get the runtime configuration, merging with defaults if provided"""
    if runtime_config is None:
        return config["runtime"]["default"]
    else:
        cfg = json.loads(runtime_config)
        return nested_update(config["runtime"]["default"], cfg)

def run(file_paths: List[str], runtime_config: Optional[str] = None):
    """Generate tag files from a list of video/image files and a runtime config"""
    cfg = get_runtime_config(runtime_config)
    model = CaptionModel(config["weights"], config=cfg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    default_tag(model, file_paths, out_path)

def get_tag_fn(runtime_config: Optional[str] = None):
    """Create a tag function with the specified configuration"""
    cfg = get_runtime_config(runtime_config)
    model = CaptionModel(config["weights"], config=cfg)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tags')
    
    def tag_fn(file_paths: List[str]):
        default_tag(model, file_paths, out_path)
    
    return tag_fn
        
if __name__ == '__main__':
    setproctitle.setproctitle("model-caption")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('file_paths', nargs='*', type=str, help='Input file paths', default=[])
    parser.add_argument('--config', type=str, required=False, help='Runtime configuration JSON')
    parser.add_argument('--live', action='store_true', help='Run in live mode (read files from stdin)')
    
    args = parser.parse_args()
    
    if args.live:
        tag_fn = get_tag_fn(args.config)
        run_live_mode(tag_fn)
    else:
        run(args.file_paths, args.config)
