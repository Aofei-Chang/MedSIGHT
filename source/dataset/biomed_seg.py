import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# class BiomedSegDataset(Dataset):
#     def __init__(self, root, mask_root, json_path, meta_file=None, transform=None, mask_transform=None, image_size=256, clip_preprocess=None, use_semantic=False):
#         self.root = root
#         self.mask_root = mask_root
#         self.transform = transform
#         self.mask_transform = mask_transform
#         self.image_size = image_size
#         self.clip_preprocess = clip_preprocess

#         self.use_semantic = use_semantic

#         with open(json_path, 'r') as f:
#             self.data = json.load(f)
#         self.annotations = self.data['annotations']
#         self.categories = self.data['categories']
#         self.num_classes = len(self.categories)

#         # Build mapping from image_id to all masks and info
#         self.imgid_to_anns = {}
#         for ann in self.annotations:
#             img_id = ann['image_id']
#             if img_id not in self.imgid_to_anns:
#                 self.imgid_to_anns[img_id] = []
#             self.imgid_to_anns[img_id].append(ann)
#         print(f"Total images: {len(self.imgid_to_anns)}, total annotations: {len(self.annotations)}")
#         # filter annotations to include more than 6 masks
#         self.annotations = [ann for ann in self.annotations if len(self.imgid_to_anns[ann['image_id']]) > 7 and len(self.imgid_to_anns[ann['image_id']]) <= 16] #6
#         print(f"Filtered annotations: {len(self.annotations)}")
#         # Build image list (unique image_id)
#         self.image_infos = {}
#         for ann in self.annotations:
#             img_id = ann['image_id']
#             if img_id not in self.image_infos:
#                 self.image_infos[img_id] = ann['file_name']
#         self.image_ids = list(self.image_infos.keys())

#     def __len__(self):
#         return len(self.image_ids)

#     def __getitem__(self, idx):
#         img_id = self.image_ids[idx]
#         img_file = self.image_infos[img_id]
#         img_path = os.path.join(self.root, img_file)
#         image = self.clip_preprocess(Image.open(img_path)).unsqueeze(0)

#         # Get all masks for this image
#         anns = self.imgid_to_anns[img_id]
#         mask_list = []
#         bbox_list = []
#         category_list = []
#         sentences_list = []
#         for ann in anns:
#             mask_file = ann['mask_file']
#             mask_path = os.path.join(self.mask_root, mask_file)
#             mask = Image.open(mask_path).convert('L')
#             # if self.mask_transform:
#             #     mask = self.mask_transform(mask)
#             # else:
#             mask = T.ToTensor()(mask)
#             # Binarize mask
#             mask = (mask > 0.5).float()
#             mask_list.append(mask)
#             bbox_list.append(torch.tensor(ann['bbox'], dtype=torch.float32))
#             category_list.append(ann['category_id'])
#             sentences_list.append([s['sent'] for s in ann.get('sentences', [])])

#         # Stack masks to (num_gt, H, W)
#         if len(mask_list) > 0:
#             masks = torch.stack(mask_list, dim=0)
#         else:
#             masks = torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
#         bboxes = torch.stack(bbox_list, dim=0) if bbox_list else torch.zeros((0, 4), dtype=torch.float32)
#         categories = torch.tensor(category_list, dtype=torch.long) if category_list else torch.zeros((0,), dtype=torch.long)
#         # For classification: return category labels for each mask
#         class_labels = categories  # (num_gt,)
#         return image, masks, class_labels, None
    


class BiomedSegDataset(Dataset):
    def __init__(self, root, mask_root, json_path, meta_file=None, transform=None, mask_transform=None, use_semantic=False, image_size=256, clip_preprocess=None, text_encoder=None, text_tokenizer=None):
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_size = image_size
        self.clip_preprocess = clip_preprocess
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.use_semantic = use_semantic
        self.device = None

        # If meta_file is provided, load multiple datasets
        if meta_file is not None:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            roots = meta['roots']
            mask_roots = meta['mask_roots']
            json_paths = meta['json_paths']
            assert len(roots) == len(mask_roots) == len(json_paths)
        else:
            roots = [root]
            mask_roots = [mask_root]
            json_paths = [json_path]

        self.imgid_to_anns = {}
        self.image_infos = {}
        self.categories = None
        self.num_classes = None
        all_annotations = []
        for r, m, j in zip(roots, mask_roots, json_paths):
            with open(j, 'r') as f:
                data = json.load(f)
            annotations = data['annotations']
            categories = data['categories']
            if self.categories is None:
                self.categories = categories
                self.num_classes = len(categories)
            # Build mapping from image_id to all masks and info
            for ann in annotations:
                img_id = ann['image_id']
                # Make img_id unique across datasets by prefixing with dataset index
                unique_img_id = f"{r}_{img_id}"
                ann['image_id'] = unique_img_id
                ann['file_name'] = os.path.join(r, ann['file_name'])
                ann['mask_file'] = os.path.join(m, ann['mask_file'])
                if unique_img_id not in self.imgid_to_anns:
                    self.imgid_to_anns[unique_img_id] = []
                self.imgid_to_anns[unique_img_id].append(ann)
            all_annotations.extend(annotations)
            for ann in annotations:
                img_id = ann['image_id']
                if img_id not in self.image_infos:
                    self.image_infos[img_id] = ann['file_name']

        print(f"Total images: {len(self.imgid_to_anns)}, total annotations: {len(all_annotations)}")
        # filter annotations to include more than 0 masks
        filtered_annotations = [ann for ann in all_annotations if len(self.imgid_to_anns[ann['image_id']]) > 0 and len(self.imgid_to_anns[ann['image_id']]) <= 16]
        print(f"Filtered annotations: {len(filtered_annotations)}")
        self.annotations = filtered_annotations
        self.image_ids = list(self.image_infos.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_file = self.image_infos[img_id]
        image = self.clip_preprocess(Image.open(img_file)).unsqueeze(0)

        # Get all masks for this image
        anns = self.imgid_to_anns[img_id]
        mask_list = []
        bbox_list = []
        category_list = []
        sentences_list = []
        for ann in anns:
            mask = Image.open(ann['mask_file']).convert('L')
            mask = T.ToTensor()(mask)
            mask = (mask > 0.5).float()
            mask_list.append(mask)
            bbox_list.append(torch.tensor(ann['bbox'], dtype=torch.float32))
            category_list.append(ann['category_id'])
            sentences_list.append([s['sent'] for s in ann.get('sentences', [])])

        if len(mask_list) > 0:
            masks = torch.stack(mask_list, dim=0)
        else:
            masks = torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
        bboxes = torch.stack(bbox_list, dim=0) if bbox_list else torch.zeros((0, 4), dtype=torch.float32)
        categories = torch.tensor(category_list, dtype=torch.long) if category_list else torch.zeros((0,), dtype=torch.long)
        class_labels = categories

        # Encode sentences if text_encoder is provided
        text_embeddings = None
        if self.use_semantic and self.text_encoder is not None:
            # print("encoding text descriptions")
            # Select a random sentence per mask (if available), else empty string
            random_sentences = [
                random.choice(sent_list) if sent_list else "" for sent_list in sentences_list
            ]
            texts = [self.text_tokenizer(cls_text).to(self.device, non_blocking=True) for cls_text in random_sentences]
            texts = torch.cat(texts, dim=0)
            text_embeddings = self.text_encoder(texts).to("cpu") # shape [num_masks, dimension]

        return image, masks, class_labels, text_embeddings