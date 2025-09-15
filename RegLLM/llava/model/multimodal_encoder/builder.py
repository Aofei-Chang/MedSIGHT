import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, UnimedCLIPVisionTower
from .vision_tokenizer import VQTower
from .siglip_encoder import SigLipVisionTower
from llava.mm_utils import VQType

from .open_clip import create_model_and_transforms, get_mean_std

# import sys
# tokenizer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../source"))
# if tokenizer_dir not in sys.path:
#     sys.path.insert(0, tokenizer_dir)

# from tokenizer.vq_model import VQ_models

def build_vision_tower(vision_tower_cfg, **kwargs):

    mm_vision_vq_type = getattr(vision_tower_cfg, 'mm_vision_vq_type', VQType.CLIP)
    if isinstance(mm_vision_vq_type, str):
        mm_vision_vq_type = vision_tower_cfg.mm_vision_vq_type = VQType[mm_vision_vq_type]
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)

    # if "regtok" in vision_tower:
    #     return None # RegTok vision tower, intialize it with specific args in another process

    # if "unimed_clip" in vision_tower:
    #     model_name = 'ViT-L-14-336-quickgelu' # available pretrained weights ['ViT-L-14-336-quickgelu', 'ViT-B-16-quickgelu']
    #     pretrained_weights = vision_tower # Path to pretrained weights
    #     text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" # available pretrained weights ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"]
    #     mean, std = get_mean_std()
    #     model, _, preprocess = create_model_and_transforms(
    #         model_name,
    #         pretrained_weights,   
    #         precision='amp',
    #         force_quick_gelu=True,
    #         mean=mean, std=std,
    #         inmem=True,
    #         text_encoder_name=text_encoder_name,)
    #     return UnimedCLIPVisionTower(vision_tower=model.visual, preprocess_func=preprocess, args=vision_tower_cfg, **kwargs)
    # self.image_encoder = model.visual.cuda()



    if mm_vision_vq_type == VQType.CLIP or vision_tower.startswith("openai"):
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
            if use_s2:
                return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
            else:
                return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif mm_vision_vq_type in [VQType.RegTok, 
                               VQType.OPEN_CLIP]:
        if is_absolute_path_exists:
            return VQTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
