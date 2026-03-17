import pytest
import numpy as np

from run import CaptionModel

def test_model():
    model = CaptionModel(weights="models/caption/git-large-textcaps")
    img = (255 * np.random.rand(224, 224, 3)).astype(np.uint8)
    tags = model.tag_frame(img)
    assert len(tags) == 1
    assert isinstance(tags[0].tag, str)