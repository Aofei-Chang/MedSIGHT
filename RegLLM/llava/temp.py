# ...existing code...

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
        return base_out

    @torch.no_grad()
    def merge_into_base(self, unfreeze_base: bool = True) -> nn.Embedding:
        """
        Copy trained codebook_emb rows into base_emb at positions given by codebook_ids.
        Returns the updated base_emb (now containing codebook weights).
        Call this after loading checkpoint but before training.
        """
        # Handle DeepSpeed ZeRO-3 parameter gathering
        try:
            from deepspeed import zero as ds_zero
            gather_base = ds_zero.GatheredParameters(list(self.base_emb.parameters()), modifier_rank=None)
            gather_code = ds_zero.GatheredParameters(list(self.codebook_emb.parameters()), modifier_rank=None)
        except Exception:
            # Fallback for non-DS environments
            class _Noop:
                def __enter__(self): return self
                def __exit__(self, *args): return False
            gather_base = _Noop()
            gather_code = _Noop()

        with gather_base, gather_code:
            base_w = self.base_emb.weight
            code_w = self.codebook_emb.weight
            
            # Type and device alignment
            if base_w.dtype != code_w.dtype:
                code_w = code_w.to(base_w.dtype)
            code_ids = self.codebook_ids.to(base_w.device)
            
            # Sanity check
            if code_ids.max().item() >= base_w.shape[0]:
                raise ValueError(
                    f"Codebook id {code_ids.max().item()} >= base vocab size {base_w.shape[0]}. "
                    "Did you resize_token_embeddings?"
                )
            
            # Copy codebook rows into base embedding
            base_w.index_copy_(0, code_ids, code_w.to(base_w.device))
            print(f"Merged {len(code_ids)} codebook embeddings into base at indices {code_ids[:5].tolist()}...")

        # Optionally unfreeze for full finetuning
        for p in self.base_emb.parameters():
            p.requires_grad = unfreeze_base
        
        if unfreeze_base:
            print("Base embedding unfrozen for full finetuning.")
        
        return self.base_emb

# ...existing code...

class RegSegForCausalLM(LlavaQwenForCausalLM):
    def __init__(self, config, **kwargs):
        seg_token_ids = kwargs.pop("seg_token_ids", None)
        self.use_seg_loss = kwargs.get("use_seg_loss", False)
        self.codebook_token_ids = seg_token_ids
        self.train_all_embeddings = kwargs.pop("train_all_embeddings", False)
        self.load_codebook_embeddings = kwargs.pop("load_codebook_embeddings", False)
        
        # Flag to merge codebook into base after loading checkpoint
        self.merge_codebook_for_full_ft = kwargs.pop("merge_codebook_for_full_ft", False)
        
        config.use_sep_proj = kwargs.pop("use_sep_proj", False) 
        super().__init__(config)
        self.model = RegSegModel(config, seg_token_ids=seg_token_ids, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        
        # Initialize SplitEmbedding structure (before loading weights)
        if not self.train_all_embeddings and self.load_codebook_embeddings:
            self._init_codebook()
    
    def _init_codebook(self):
        """Create SplitEmbedding wrapper (called before from_pretrained loads weights)."""
        base_emb = self.get_input_embeddings()
        split_emb = SplitEmbedding(base_emb, self.codebook_token_ids)
        self.set_input_embeddings(split_emb)
        print(f"SplitEmbedding created: base_emb {base_emb.weight.shape} frozen, "
              f"{len(self.codebook_token_ids)} codebook tokens trainable.")

    @torch.no_grad()
    def merge_codebook_into_base(self, unfreeze_base: bool = True, remove_split: bool = True):
        """
        Merge trained codebook_emb into base_emb and optionally remove SplitEmbedding.
        Call this AFTER from_pretrained has loaded checkpoint weights.
        
        Args:
            unfreeze_base: If True, make base_emb trainable for full finetuning
            remove_split: If True, replace SplitEmbedding with plain base_emb
        """
        emb = self.get_input_embeddings()
        
        if not isinstance(emb, SplitEmbedding):
            print("No SplitEmbedding found; nothing to merge.")
            return
        
        # Merge codebook weights into base
        base = emb.merge_into_base(unfreeze_base=unfreeze_base)
        
        if remove_split:
            # Replace input embeddings with the unified base embedding
            self.set_input_embeddings(base)
            print("Removed SplitEmbedding wrapper; using unified base embedding.")
        else:
            print("Kept SplitEmbedding wrapper (merged weights remain in base_emb).")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Override to support auto-merge after loading checkpoint.
        Pass merge_codebook_for_full_ft=True to auto-merge and unfreeze.
        """
        # Extract our custom flag before super() sees it
        merge_flag = kwargs.pop("merge_codebook_for_full_ft", False)
        
        # Load model normally (SplitEmbedding.codebook_emb will be loaded from checkpoint)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Auto-merge if requested
        if merge_flag:
            model.merge_codebook_into_base(unfreeze_base=True, remove_split=True)
        
        return model

# ...existing code...