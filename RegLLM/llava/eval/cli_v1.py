import sys
import os
# file_path = os.path.abspath(__file__)
# dir_path = os.path.dirname(file_path)
# print(dir_path)
# sys.path.insert(0, dir_path)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model import *

from RegLLM.RegSeg import RegSegForCausalLM

from transformers import AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import torch
import torch.nn as nn
import copy

from PIL import Image

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', "mask_decoder", "region", 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            name_to_Add = names[0] if len(names) == 1 else names[-1]
            if len(name_to_Add) > 1:
                lora_module_names.add(name_to_Add)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class SplitEmbedding(nn.Module):
    def __init__(self, base_emb: nn.Embedding, codebook_token_ids: list[int]):
        super().__init__()
        self.base_emb = base_emb
        for p in self.base_emb.parameters():
            p.requires_grad = False

        self.codebook_ids = torch.tensor(codebook_token_ids, dtype=torch.long)
        self.id_to_local = {tid.item(): i for i, tid in enumerate(self.codebook_ids)}
        self.codebook_emb = nn.Embedding(len(codebook_token_ids),
                                         base_emb.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base_out = self.base_emb(input_ids)

        # Prepare idx_map (mapping from global token id → local codebook id)
        idx_map = torch.full_like(input_ids, -1, dtype=torch.long)
        for i, tid in enumerate(self.codebook_ids):
            idx_map[input_ids == tid] = i

        # mask where codebook tokens appear
        mask = idx_map >= 0

        if mask.any():
            # Compute codebook embeddings for masked positions
            code_emb = self.codebook_emb(idx_map.clamp(min=0))
            # Create a new tensor: where mask True → use codebook_emb, else → base_out
            out = torch.where(mask.unsqueeze(-1), code_emb, base_out)
        else:
            out = base_out

        return out

class RegLLMChatbot():
    def __init__(self, model_dir, pretrained_model_dir=None, peft_path=None, model_args=None, device = 'cuda'):
        self.model_dir = model_dir
        self.pretrained_model_dir = pretrained_model_dir
        self.peft_path = peft_path

        self.gen_kwargs = {
            'do_sample': True,
            'max_new_tokens': 512,
            'min_new_tokens': 1,
            'temperature': .2,
            'repetition_penalty': 1.2
        }
        # self.gen_kwargs = {
        #     'do_sample': True,
        #     'max_new_tokens': 512,
        #     'temperature': 0.7,              # balanced creativity
        #     'top_p': 0.9,                    # nucleus sampling
        #     'top_k': 50,                     # limit to top 50 tokens
        #     'repetition_penalty': 1.1,
        #     'length_penalty': 1.0,
        # }
        self.device = device
        
        self.history = []
        self.images = []
        self.debug = True
        self.max_image_num = 6
        if model_args is None:
            self.model_args = {
                "model_name_or_path": self.model_dir,
                "pretrained_llm_path": self.pretrained_model_dir,
                "regtok_config_path": "/qumulo/shared_data/aofei_summer/RegTok/source/tokenizer/regtok_config.yaml",
                "regtok_weight_path": "/qumulo/shared_data/aofei_summer/intern_records/RegTok/checkpoints/RegTok_pipeline_full_wo_quant/002-RegTok/checkpoints/0079280.pt",
                "use_regtok": True,
                "mm_vision_vq_type": "RegTok",
                "use_region_tokens": False,
                "vision_tower": "/qumulo/shared_data/aofei_summer/CLIPs/unimed_clip_vit_l14.pt",
                "mm_use_im_start_end": False,
                "mm_use_im_patch_token": True,
                "mm_vision_select_feature": "patch",
                "mm_patch_merge_type": "flat",
                "mm_projector_type": "mlp2x_gelu",
                "pretrain_mm_mlp_adapter": None,
                "mm_vision_select_layer": -1,

            }
        else:
            self.model_args = model_args
        from types import SimpleNamespace
        self.model_args = SimpleNamespace(**self.model_args)
        self.training_args = {
            
        }
        self.init_components()

    def init_components(self):
        model_args = self.model_args
        training_args = self.training_args
        d = self.model_dir
        print(f'loading model from {self.model_dir}')
        from llava.model.language_model.llava_qwen3 import LlavaQwenForCausalLM
        # model = LlavaQwenForCausalLM.from_pretrained(
        #             "Qwen/Qwen3-8B",
        #             # cache_dir=training_args.cache_dir,
        #             # attn_implementation=attn_implementation,
        #             torch_dtype=torch.bfloat16 ,
        #             low_cpu_mem_usage=False,
        #         )

        if not model_args.output_segmentation:
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path if model_args.pretrained_llm_path is None else model_args.pretrained_llm_path,
                # attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )
        else:
            print("use RegSegForCausalLM!")
            import json
            token_ids_path = "/data/aofei/output/MedSight/1110_full_instruct_71k_nosep/added_tokens.json"
            # token_ids_path = os.path.join(model_args.pretrained_llm_path, "added_tokens.json")
            codebook_token_ids = [i[1] for i in list(json.load(open(token_ids_path)).items()) if i[0].startswith("[")]
            print(len(codebook_token_ids), "codebook_token_ids")
            regseg_args = {
                "seg_token_ids": codebook_token_ids,
                "use_seg_loss": self.model_args.use_seg_loss,
                "train_all_embeddings": self.model_args.train_all_embeddings,
                "use_lightweight_decoder": self.model_args.use_lightweight_decoder,
                "load_codebook_embeddings": self.model_args.load_codebook_embeddings,
                # "full_inference_stage": self.model_args.full_inference_stage,
                "use_sep_proj": self.model_args.use_sep_proj,
                "decoder_dim": 1024,
                "ce_loss_weight": 1.0,
                "mask_loss_weight": 1.0,
                "dice_loss_weight": 1.0,
                "bce_loss_weight": 1.0,
            }
            model = RegSegForCausalLM.from_pretrained(
                # model_args.model_name_or_path,
                model_args.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                **regseg_args,
            )
            # print("codebook", model.model.embed_tokens.codebook_emb.weight)
        
        # if model_args.vision_tower is not None:
        print("load vision tower!")
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            # fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        # print(vision_tower, "vision tower")
        vision_tower.to(dtype=torch.bfloat16, device="cuda")
        print("pre loading complete!")
        # model.config.image_aspect_ratio = data_args.image_aspect_ratio
        # model.config.tokenizer_padding_side = tokenizer.padding_side
        # model.config.tokenizer_model_max_length = tokenizer.model_max_length

        
        from safetensors.torch import load_file
        import glob

        # Path to your model directory
        # model_dir = "/qumulo/shared_data/aofei_summer/RegTok/RegLLM/checkpoints/i2t_instruct"
        model_dir = self.model_dir
        if model_dir is not None and not model_args.output_segmentation:
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
            self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id

            # model.resize_token_embeddings(len(tokenizer))

            print(f"loading from {model_dir}")
            # Find all safetensors shards
            safetensor_files = sorted(glob.glob(f"{model_dir}/*.safetensors"))

            # Merge all state_dicts
            state_dict = {}
            for file in safetensor_files:
                state_dict.update(load_file(file))

            # Now load into your model
            result = model.load_state_dict(state_dict, strict=False)
            print(result.missing_keys, "missing:", result.unexpected_keys)


        elif model_args.output_segmentation and model_args.pretrained_llm_path is not None:
            print("Loading tokenizer from", model_args.pretrained_llm_path)
            if model_args.tokenizer_path is not None:
                print("Loading tokenizer from", model_args.tokenizer_path)
                tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_llm_path)
            #-------------
            # # weight_path = self.model_dir
            # # model_state_dicts = torch.load(weight_path)
            # model, loading_info = LlavaQwenForCausalLM.from_pretrained(self.model_dir, torch_dtype=torch.bfloat16)
            # missing_keys = loading_info['missing_keys'] # keys exists in model architecture but does not exist in ckpt
            # unexpected_keys = loading_info['unexpected_keys'] # keys exists in ckpt but are not loaded by the model 
            # assert all(['vision_tower' in k for k in unexpected_keys])
                        # vision_tower = model.get_vision_tower()

            # if model_args.vision_tower is not None:
            #     print("load vision tower!")
            #     model.get_model().initialize_vision_modules(
            #         model_args=model_args,
            #         fsdp=training_args.fsdp
            #     )

            # vision_tower = model.get_vision_tower()
            # # print(vision_tower, "vision tower")
            # vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            # data_args.image_processor = vision_tower.image_processor
            # data_args.is_multimodal = True

            # if not vision_tower.is_loaded:
            #     vision_tower.load_model()
            #     vision_tower.vision_tower = vision_tower.vision_tower.from_pretrained(self.model_dir)
        elif self.peft_path is not None:
            peft_path = self.peft_path
            from peft import PeftModel
            print(f"Loading LoRA weights from {peft_path}")
            model = PeftModel.from_pretrained(model, peft_path)
            print(f"Merging weights")
            model = model.merge_and_unload()

            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
            self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            # load alignment layer
            new_state_dict = {}
            align_path = os.path.join(peft_path, "non_lora_trainables.bin")
            align_state_dict = torch.load(align_path, map_location='cuda')
            for key, value in align_state_dict.items():
                # Replace "base_model.model" with an empty string to remove it
                new_key = key.replace("base_model.model", "")
                if new_key.startswith("."):
                    new_key = new_key[1:]
                new_state_dict[new_key] = value.to("cuda")
                # new_state_dict[new_key] = value
            result = model.load_state_dict(new_state_dict, strict=False)
            print("Unexpected peft weights:", result.unexpected_keys)
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id
            self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        self.tokenizer = tokenizer

        # load peft weights and non-lora modules
        if model_args.peft_path is not None:
            peft_path = model_args.peft_path
            if model_args.resize_embedding:
                model.resize_token_embeddings(len(tokenizer))
                print("Current vocab size & model embedding size: ", len(tokenizer), model.get_input_embeddings().weight.size())

            if not model_args.train_all_embeddings and type(model.get_input_embeddings()) is nn.Embedding:
                base_emb = model.get_input_embeddings()
                split_emb = SplitEmbedding(base_emb, codebook_token_ids)
                model.set_input_embeddings(split_emb)

                print(f"SplitEmbedding created: base_emb frozen, "
                    f"{len(codebook_token_ids)} codebook tokens trainable.")
        

            print(f"Loading LoRA weights from {peft_path}")
            if model_args.use_moe:
                from llava.peft import LoraConfig, get_peft_model, PeftModel
                lora_config = LoraConfig(
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    target_modules=find_all_linear_names(model),
                    lora_dropout=model_args.lora_dropout,
                    bias='none',
                    task_type="CAUSAL_LM",
                    lora_nums=model_args.lora_nums,
                )
                model = get_peft_model(model, lora_config)
                hlora_weights = torch.load(os.path.join(model_args.peft_path, "adapter_model.bin"))
                hlora_unexpected_keys = model.load_state_dict(hlora_weights, strict=False)[1]
                if hlora_unexpected_keys:
                    print(f"Warning: Unexpected keys in hlora checkpoint: {hlora_unexpected_keys}")
                
                
            else:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, peft_path)
                print(f"Merging weights")
                model = model.merge_and_unload()

            if os.path.exists(os.path.join(peft_path, "non_lora_trainables.bin")):
                non_lora_trainables_path = os.path.join(peft_path, "non_lora_trainables.bin")
                non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cuda')
                print(f"Loading non-LoRA weights from {non_lora_trainables_path}")
                print(non_lora_trainables.keys())
                new_state_dict = {}
                for key, value in non_lora_trainables.items():
                    new_key = key
                    if not model_args.use_moe:
                        new_key = key.replace("base_model.model", "")
                        if new_key.startswith("."):
                            new_key = new_key[1:]
                    new_state_dict[new_key] = value.to("cuda")
                    print("New state dict keys:", new_state_dict.keys())
                    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                    print("Unexpected keys:", unexpected)

            # non_lora_trainables_path = os.path.join(peft_path, "non_lora_trainables.bin")
            # non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cuda')
            # print(f"Loading non-LoRA weights from {non_lora_trainables_path}")
            # print(non_lora_trainables.keys())
            # new_state_dict = {}
            # for key, value in non_lora_trainables.items():
            #     # Replace "base_model.model" with an empty string to remove it
            #     new_key = key
            #     if not model_args.use_moe:
            #         new_key = key.replace("base_model.model", "")
            #         if new_key.startswith("."):
            #             new_key = new_key[1:]
            #     new_state_dict[new_key] = value.to("cuda")
            # print("New state dict keys:", new_state_dict.keys())
            # missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            # print("Unexpected keys:", unexpected)

        vision_tower.to(dtype=torch.bfloat16, device=model.device)
        image_processor = vision_tower.image_processor
        self.processor = image_processor
        model.eval()
        self.model = model.to(self.device).to(torch.bfloat16)
        # self.model.config.tokenizer_padding_side = 'left'


    def clear_history(self,):
        self.images = []
        self.history = []

    def tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None): # copied from llava
        prompt_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def preprocess(self, data: list, return_tensors='pt'):
        '''
        [
            {
                'from': 'human',
                'value': xxx,
            },
            {
                'from': 'gpt',
                'value': xxx
            }
        ]
        '''
        if not isinstance(data, list):
            raise ValueError('must be a list')        
        return self.preprocess_qwen(data, return_tensors=return_tensors)
    
    def preprocess_qwen(self, sources, tokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant."):
        # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        roles = {"human": "user", "gpt": "assistant"}

        # Add image tokens to tokenizer as a special tokens
        # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
        tokenizer = copy.deepcopy(tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        for token_name, token_id in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids):
            if "im_start" in token_name:
                im_start = token_id
            elif "im_end" in token_name:
                im_end = token_id
        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids

        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # _system = tokenizer("system").input_ids + nl_tokens
        # _user = tokenizer("user").input_ids + nl_tokens
        # _assistant = tokenizer("assistant").input_ids + nl_tokens

        # Apply prompt templates
        input_ids = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []

            # New version, use apply chat template
            # Build system message for each sentence
            input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                # print(conv, "conversations")
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id

            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids
    
    def preprocess_huatuo(self, convs: list, return_tensors) -> list: # tokenize and concat the coversations
        input_ids = None
        convs = [ conv for conv in convs if conv['value'] is not None]
        round_num = len(convs)//2

        for ind in range(round_num):
            h = convs[ind*2]['value'].strip()
            h = f"<|user|>\n{h}\n" 

            g = convs[ind*2+1]['value']
            g = f"<|assistant|>\n{g} \n"

            cur_input_ids = self.tokenizer_image_token(prompt=h, return_tensors=return_tensors)

            if input_ids is None:
                input_ids = cur_input_ids
            else:
                input_ids = torch.cat([input_ids, cur_input_ids])
            
            cur_input_ids = self.tokenizer(g, add_special_tokens= False, truncation=True, return_tensors='pt').input_ids[0]
            input_ids = torch.cat([input_ids, cur_input_ids])
        
        h = convs[-1]['value'].strip()
        h = f"<|user|>\n{h}\n<|assistant|>\n"
        cur_input_ids = self.tokenizer_image_token(prompt=h, return_tensors=return_tensors)

        if input_ids is None:
            input_ids = cur_input_ids
        else:
            input_ids = torch.cat([input_ids, cur_input_ids])
        
        if self.debug:
            self.debug = False

        return input_ids


    def input_moderation(self, t: str):
        blacklist = ['<image>']
        for b in blacklist:
            t = t.replace(b, '')
        return t
    
    def insert_image_placeholder(self, t, num_images, placeholder='<image>', sep='\n'):
        for _ in range(num_images):
            t = f"{placeholder}{sep}" + t

        return t
    
    def get_conv(self, text):
        ret = []
        if self.history is None:
            self.history = []
        
        for conv in self.history:
            ret.append({'from': 'human', 'value': conv[0]})
            ret.append({'from': 'gpt', 'value': conv[1]})

        ret.append({'from': 'human', 'value': text})
        ret.append({'from': 'gpt', 'value': None})

        return ret

    def get_conv_without_history(self, text):
        ret = []

        ret.append({'from': 'human', 'value': text})
        # ret.append({'from': 'gpt', 'value': ""})

        return ret
    
    def get_image_tensors(self, images):
        list_image_tensors = []
        # crop_size = self.processor.crop_size
        processor = self.processor
        for fp in images:
            if fp is None: # None is used as a placeholder
                continue
            elif isinstance(fp, str):
                image = Image.open(fp).convert('RGB')
            elif isinstance(fp, Image.Image):
                image = fp # already an image
            else:
                raise TypeError(f'Unsupported type {type(fp)}')

            # if False or self.data_args.image_aspect_ratio == 'pad':
            #     def expand2square(pil_img, background_color):
            #         width, height = pil_img.size
            #         if width == height:
            #             return pil_img
            #         elif width > height:
            #             result = Image.new(pil_img.mode, (width, width), background_color)
            #             result.paste(pil_img, (0, (width - height) // 2))
            #             return result
            #         else:
            #             result = Image.new(pil_img.mode, (height, height), background_color)
            #             result.paste(pil_img, ((height - width) // 2, 0))
            #             return result
            #     image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            #     image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0] # a tensor
            list_image_tensors.append(image.to(self.device))
        # if len(list_image_tensors) == 0:
        #     list_image_tensors.append(torch.zeros(3, crop_size['height'], crop_size['width']).to(self.device))
        return list_image_tensors

    def inference(self, text, images=None, output_seg=False):
        '''
        text: str
        images: list[str]
        '''
        
        # image
        if images is None:
            images = []

        # if isinstance(images,str):
        images = [images]

        valid_images = []
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB') # make sure that the path exists
                valid_images.append(img)
            except:
                print(f'{img} This image is wrong.')
                continue
        images = valid_images
        if len(valid_images) > self.max_image_num:
            images = images[:self.max_image_num]

        # text
        # sources = preprocess_multimodal(
        #         copy.deepcopy([e["conversations"] for e in sources]),
        #         self.data_args)
        text = self.input_moderation(text)
        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)

        conv = self.get_conv_without_history(text)
        input_ids = self.preprocess_qwen([conv], tokenizer=self.tokenizer, has_image=True).to(self.device)
        # print(input_ids, "input_idss")
        if len(images) > 0:
            list_image_tensors = self.get_image_tensors(images)
            image_tensors = torch.stack(list_image_tensors).to(dtype=torch.bfloat16).to(self.device)
        else:
            image_tensors = None

        with torch.inference_mode():
            # print(input_ids, image_tensors.size(), "input")
            if output_seg:
                attn_mask = self.gen_kwargs.pop('attention_mask', None)
                self.gen_kwargs['do_sample'] = False
                self.gen_kwargs['temperature'] = 1.0
                self.gen_kwargs['top_p'] = 1.0
                self.gen_kwargs['top_k'] = 50
                outputs = self.model.generate(
                    input_ids,
                    images=image_tensors.to(torch.bfloat16),
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    **self.gen_kwargs)
                output_ids = outputs.sequences

                final_layer_token_states = [step[-1] for step in outputs.hidden_states]
                final_layer_token_states[0] = final_layer_token_states[0][:, -1, :].unsqueeze(1)
                final_layer_token_states = torch.stack(final_layer_token_states, dim=1).squeeze(-2)

                print(final_layer_token_states.shape, output_ids.shape)
                seg_embeddings = self.model.model.collect_seg_token_embeddings(final_layer_token_states, output_ids)
                region_codes = self.model.model.token_projection(
                    seg_embeddings.view(-1, seg_embeddings.size(-1))
                ).view(seg_embeddings.shape[0], seg_embeddings.shape[1], -1)
                # image_features = self.model.model.extract_multi_visual_features(image_tensors.to(torch.bfloat16).to(final_layer_token_states.dtype))
                seg_logits, class_logits = self.model.model.decode_masks(region_codes)
                
                # outputs = self.model.evaluate(input_ids=input_ids, images=image_tensors)
                # output_id = outputs.sequences
                # # hidden_states = outputs.hidden_states[-1]
                # hidden_states_last_layer_tokens = [i[-1] for i in outputs.hidden_states]
                # hidden_states = torch.cat(hidden_states_last_layer_tokens, dim=1)
                # print(hidden_states.shape, "hidden states", output_id.size())

                # # print(len(hidden_states), output_id.size(), len(outputs.hidden_states), hidden_states[0].shape)
                # seg_embeddings = self.model.model.collect_seg_token_embeddings(hidden_states, output_id)
                # region_codes = self.model.model.token_projection(
                #     seg_embeddings.view(-1, seg_embeddings.size(-1))
                # ).view(seg_embeddings.shape[0], seg_embeddings.shape[1], -1)
                # image_features = self.model.model.extract_visual_features(image_tensors.to(hidden_states.dtype))
                # seg_logits, class_logits = self.model.model.decode_masks(region_codes, image_features)

                answers = []
                for output_id in output_ids:
                    answers.append(self.tokenizer.decode(output_id, skip_special_tokens=True).strip())
                return {
                    "answers": answers,
                    "mask_logits": seg_logits
                }
            else:
                # self.gen_kwargs['do_sample'] = False
                # self.gen_kwargs['temperature'] = 1.0
                # self.gen_kwargs['top_p'] = 1.0
                # self.gen_kwargs['top_k'] = 50
                attn_mask = self.gen_kwargs.pop('attention_mask', None)
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensors.to(torch.bfloat16),
                    use_cache=True,
                    **self.gen_kwargs)
                answers = []
                for output_id in output_ids:
                    answers.append(self.tokenizer.decode(output_id, skip_special_tokens=True).strip())
                return answers, output_ids

    def chat(self, text: str, images: list[str]=None, ):
        '''
        images: list[str], images for this round
        text: str
        '''
        text = self.input_moderation(text)
        if text == '':
            return 'Please type in something'

        if isinstance(images, str) or isinstance(images, Image.Image):
            images = [images]
        
        valid_images = []
        if images is None:
            images = []
        
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB') # make sure that the path exists
                valid_images.append(img)
            except:
                continue

        images = valid_images

        self.images.extend(images)


        assert len(images) < self.max_image_num, f'at most {self.max_image_num} images'

        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)
        # make conv
        conv = self.get_conv(text)
        # make input ids
        input_ids = self.preprocess(conv, return_tensors='pt').unsqueeze(0).to(self.device)

        if len(self.images) > 0:
            list_image_tensors = self.get_image_tensors(self.images)
            image_tensors = torch.stack(list_image_tensors)
        else:
            image_tensors = None

        streamer = TextIteratorStreamer(self.tokenizer,skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs=input_ids,images=image_tensors.to(dtype=torch.bfloat16) if image_tensors is not None else image_tensors, streamer=streamer,use_cache=True,**self.gen_kwargs)


        with torch.inference_mode():
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            generated_text = ''
            sep = self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)
            for new_text in streamer:
                if sep in new_text:
                    new_text = self.remove_overlap(generated_text,new_text[:-len(sep)])
                    for char in new_text:
                        generated_text += char
                        print(char,end='',flush = True)
                    break
                for char in new_text:
                    generated_text += char
                    print(char,end='',flush = True)
        answer = generated_text

        self.history.append([text, answer])

        return answer


if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Args of Data Preprocess')

    parser.add_argument('--model_dir', default='', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()

    bot = RegLLMChatbot(args.model_dir, args.device)

    # test
    # print(bot.inference('what show in this picture?',['./output.png']))
    # print(bot.inference('hi'))

    while True:
        images = input('images, split by ",": ')
        images = [i.strip() for i in images.split(',') if len(i.strip()) > 1 ]
        text = input('USER ("clear" to clear history, "q" to exit): ')
        if text.lower() in ['q', 'quit']:
            exit()

        if text.lower() == 'clear':
            bot.history = []
            bot.images = []
            continue

        answer = bot.chat(images=images, text=text)

        images = None # already in the history

        print()
        print(f'GPT: {answer}')
        print()