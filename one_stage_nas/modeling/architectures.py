from .HSI_supernet import HSI_supernet
from .HSI_compnet import HSI_compnet

ARCHITECTURES = {
    "HSI_supernet": HSI_supernet,
    "HSI_compnet": HSI_compnet,
}


def build_model(cfg):
    meta_arch = ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
