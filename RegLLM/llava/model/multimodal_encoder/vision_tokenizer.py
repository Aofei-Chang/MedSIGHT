import torch
import torch.nn as nn
import yaml
from .tokenflow_vq import tokenflow_model
from .open_clip_encoder import SigLipVisionTower

import os
from PIL import Image
from typing import Optional, Tuple, Union, Dict
from functools import partial, reduce

from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers import PretrainedConfig


from einops import rearrange
from llava.mm_utils import VQType 


class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)

class UnimedCLIPProcesser():
    def __init__(self, preprocess_func, args):
        self.preprocess_func = preprocess_func
        self.args = args

    def preprocess(self, images, return_tensors='pt'):
        processed_data = self.preprocess_func(images).unsqueeze(0)
        data = {'pixel_values': processed_data}
        return BatchFeature(data=data, tensor_type=return_tensors)

class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)

class VQTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.args = args
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.vq_type = getattr(args, 'mm_vision_vq_type', VQType.OPEN_CLIP)
        # self.reg_tok_config_path = getattr(args, 'regtok_config_path', None)
        self.reg_tok_config_path = "/qumulo/shared_data/aofei_summer/RegTok/source/tokenizer/regtok_config.yaml"
        self.reg_tok_config = None
        if self.reg_tok_config_path and self.reg_tok_config_path.endswith(".yaml"):
            # load the config in .yaml file
            import yaml
            with open(self.reg_tok_config_path, 'r') as f:
                self.reg_tok_config = yaml.safe_load(f)

        if self.vq_type == VQType.TOKENFLOW or self.vq_type == VQType.RegTok:
            # for 224 input size
            self.CLIPVisionConfig = "openai/clip-vit-large-patch14"  # follow llava 1.5
        elif self.vq_type == VQType.OPEN_CLIP or self.vq_type == VQType.UNIMED_CLIP:
            self.CLIPVisionConfig = "openai/clip-vit-large-patch14"
        else:
            raise NotImplementedError()
        
        self.cfg_only = SigLipVisionConfig()

        self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        assert os.path.exists(self.vision_tower_name), "VQGAN model path is invalid: %s" % self.vision_tower_name
        self.image_processor = None
        if self.vq_type == VQType.RegTok:
            self.vision_tower = tokenflow_model('RegTok', cfg=self.reg_tok_config, pretrain_path=getattr(self.args, 'regtok_weight_path', None))
            self.image_processor = UnimedCLIPProcesser(self.vision_tower.unimed_preprocess, self.args)

        elif self.vq_type == VQType.OPEN_CLIP:
            self.vision_tower = SigLipVisionTower()

        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)
        if device_map:
            self.vision_tower = self.vision_tower.to(device_map)

        if self.image_processor is None:
            self.image_processor = SigLipImageProcessor()

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                with torch.no_grad():
                    image_feature, region_queries, info = self.vision_tower(image.to(device=self.device, dtype=self.dtype, llm_training=True).unsqueeze(0))
                inds = info[-1]

                image_feature_flat = rearrange(image_feature, 'b c h w -> b (h w) c').to(image.dtype)
                image_features.append(image_feature_flat)

        else:
            with torch.no_grad():
                # print(images.shape, "image shape")
                image_features, region_queries, info = self.vision_tower(images.to(device=self.device, dtype=self.dtype), llm_training=True)
            # print(image_features.shape, "image shape 00",)
            if len(image_features.shape) == 4:
                image_features = rearrange(image_features, 'b h w c -> b (h w) c').to(images.dtype)
            # print(image_features.shape, "image shape", region_queries.shape, "region queries")
            if self.args.use_region_tokens:
                # print(image_features.size(), "before concat", region_queries.size())
                if len(image_features.shape) == 2:
                    image_features = image_features.unsqueeze(0)
                if len(image_features.shape)>=3 and len(region_queries.shape) < 3:
                    region_queries = region_queries.unsqueeze(0)
                image_features = torch.cat([image_features, region_queries], dim=1)
                # print(image_features.size(), "after concat")
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.vision_tower.compression

    @property
    def num_patches(self):
        return (self.config.image_size // self.vision_tower.compression) ** 2

    @property
    def num_codebook_tokens(self):
        return self.vision_tower.n_embed

