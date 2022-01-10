import sys
sys.path.append("models")

import backbones
import heads
from Unet import Unet
from CloudNetp import CloudNetp
from SwinSegmenter import SwinSegmenter
from SeMask import SeMask

def build_backbone(cfg):
    """Build backbone.
    Args:
        cfg (dict): The backbone config.
    Returns:
        nn.Module: The built backbone.
    """
    assert isinstance(cfg, dict)
    assert 'type' in cfg
    return getattr(backbones, cfg.pop('type'))(**cfg)

def build_head(cfg):
    """Build a head from config.
    Args:
        cfg (dict): The head config, which
    Returns:
        nn.Module: The built head.
    """
    assert isinstance(cfg, dict)
    assert 'type' in cfg
    return getattr(heads, cfg.pop('type'))(**cfg)