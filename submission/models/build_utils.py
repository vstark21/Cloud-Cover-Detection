import backbones
import heads

def build_backbone(cfg, **kwargs):
    """Build backbone.
    Args:
        cfg (dict): The backbone config.
    Returns:
        nn.Module: The built backbone.
    """
    assert isinstance(cfg, dict)
    assert 'type' in cfg
    return getattr(backbones, cfg.pop('type'))(**cfg, **kwargs)

def build_head(cfg, **kwargs):
    """Build a head from config.
    Args:
        cfg (dict): The head config, which
    Returns:
        nn.Module: The built head.
    """
    assert isinstance(cfg, dict)
    assert 'type' in cfg
    return getattr(heads, cfg.pop('type'))(**cfg, **kwargs)
