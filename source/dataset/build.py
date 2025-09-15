
from dataset.biomed_seg import BiomedSegDataset

def build_dataset(args, **kwargs):
    if args.dataset == 'biomed_seg':
        return BiomedSegDataset(
            root=args.data_path,
            mask_root=args.segmentation_path,
            json_path=args.annotation_json,
            meta_file=args.batch_dataset_meta_file,
            transform=kwargs.get('transform', None),
            mask_transform=kwargs.get('mask_transform', None),
            image_size=args.image_size,
            clip_preprocess=kwargs.get('clip_preprocess', None),
            use_semantic=kwargs.get('use_semantic', False)
        )
    raise ValueError(f'dataset {args.dataset} is not supported')