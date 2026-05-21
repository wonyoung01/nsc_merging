from collections.abc import Sequence
from copy import deepcopy
from importlib import import_module

import torch
from peft import LoraConfig, get_peft_model
from peft.utils import ModulesToSaveWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

import wandb
from src.utils import get_base_model, get_lora_config, load_lora_model

from .sft_trainer import KernelSFTTrainer

# Tokens for LLaVA-1.5 and similar models
START_TOKENS: list[str] = [
    "ASSISTANT:",
]
END_TOKENS: list[str] = [
    "</s>",
]


def get_helper_module(cfg, key):
    data_name = cfg.data.lower() if isinstance(cfg.data, str) else cfg.data.name.lower()
    mod = import_module(f"vlm.messages.{data_name}")
    helper_func = getattr(mod, key, None)
    if helper_func is None or not callable(helper_func):
        raise ValueError(f"{key} not found in vlm.messages.{data_name}")
    return helper_func


def get_build_messages(cfg):
    return get_helper_module(cfg, "build_messages")


def get_build_meta(cfg):
    return get_helper_module(cfg, "build_meta")


def get_process_results(cfg):
    return get_helper_module(cfg, "process_results")


def get_evaluate_results(cfg):
    return get_helper_module(cfg, "evaluate_results")


def get_parent_and_attr(root, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)  # walk down the tree
    return parent, parts[-1]


@torch.no_grad()
def remove_modules_to_save(model, adapter_names=None, verbose=True):
    """
    Given a PEFT-wrapped model, remove modules that are marked to be saved.
    Typically it is for multi_modal_projector in LLaVA.
    Indeed, change multi_modal_projector to LoRA Linear module
    by making lora_A: full task vector lora_B: identity
    """
    # FIXME: Currently only considering linear_1 and linear_2 in multi_modal_projector
    #
    # First, Copy original model
    final_model = deepcopy(model)
    for mod_name, mod in model.named_modules():
        if not isinstance(mod, ModulesToSaveWrapper):
            continue

        # First copy original module of mod
        if isinstance(mod.original_module, LlavaMultiModalProjector):
            final_mod = deepcopy(mod.original_module)

            # Get base weights of linear_1 and linear_2
            base_w_1 = getattr(mod.original_module.linear_1, "weight", None)
            base_w_2 = getattr(mod.original_module.linear_2, "weight", None)

            if base_w_1 is None or base_w_1.ndim != 2 or base_w_2 is None or base_w_2.ndim != 2:
                if verbose:
                    print(f"[skip] {mod_name}: not a 2D Linear weight")
                continue

            # Feature space of linear_1 and linear_2 should match
            # Furthermore, in our case out_features_2 == in_features_2 is necessary.
            out_features_1, in_features_1 = base_w_1.shape
            out_features_2, in_features_2 = base_w_2.shape
            assert out_features_1 == in_features_2, (
                f"Dimension mismatch in {mod_name}: linear_1.out_features ({out_features_1}) != "
                f"linear_2.in_features ({in_features_2})"
            )

            # Make the LoRA Config appropriate for this dimension
            _lora_conf = LoraConfig(
                r=out_features_1,
                lora_alpha=out_features_1,
                target_modules=["linear_1", "linear_2"],
                bias="none",
            )
            peft_mod = get_peft_model(final_mod, _lora_conf)  # type: ignore
            # Delete default adapters created by get_peft_model
            peft_mod.delete_adapter("default")

            # Determine which adapters to touch
            names = list(mod.modules_to_save.keys())
            if adapter_names is not None:
                names = [n for n in names if n in set(adapter_names)]
            if not names:
                continue

            # Inject Adapter
            for an in names:
                A_lin_1 = mod.modules_to_save[an].linear_1
                A_lin_2 = mod.modules_to_save[an].linear_2
                dtype = A_lin_1.weight.data.dtype
                A_1 = A_lin_1.weight.data.to(torch.float64) - base_w_1.data.to(torch.float64)
                A_2 = A_lin_2.weight.data.to(torch.float64) - base_w_2.data.to(torch.float64)
                A_1 = A_1.to(dtype)
                A_2 = A_2.to(dtype)

                # Add LoRA adapters
                peft_mod.add_adapter(an, _lora_conf)

                # Copyt weights to added adapters
                peft_mod.linear_1.lora_A[an].weight.data.copy_(A_1)
                peft_mod.linear_1.lora_B[an].weight.data = torch.eye(
                    out_features_1, dtype=A_1.dtype, device=A_1.device
                )
                peft_mod.linear_2.lora_A[an].weight.data.copy_(A_2)
                peft_mod.linear_2.lora_B[an].weight.data = torch.eye(
                    out_features_2, dtype=A_2.dtype, device=A_2.device
                )
            # Finally, set mod to final_mod
            # Note: final_mod is not a PEFT model but with LoRA Linear modules inside
            parent, attr = get_parent_and_attr(final_model, mod_name)
            setattr(parent, attr, peft_mod)
            final_model.set_adapter(names[-1])  # set to one of the adapters
        elif isinstance(mod.original_module, torch.nn.Linear):

            class WrapLinear(torch.nn.Module):
                def __init__(self, linear: torch.nn.Linear):
                    super().__init__()
                    self.linear = linear

                def forward(self, x):
                    return self.linear(x)

            final_mod = WrapLinear(deepcopy(mod.original_module))

            # Get base weights of linear module
            base_w = getattr(mod.original_module, "weight", None)

            if base_w is None or base_w.ndim != 2:
                if verbose:
                    print(f"[skip] {mod_name}: not a 2D Linear weight")
                continue

            out_features, in_features = base_w.shape

            # Make the LoRA Config appropriate for this dimension
            _lora_conf = LoraConfig(
                r=out_features,
                lora_alpha=out_features,
                target_modules=["linear"],
                bias="none",
            )
            peft_mod = get_peft_model(final_mod, _lora_conf)  # type: ignore
            # Delete default adapters created by get_peft_model
            peft_mod.delete_adapter("default")

            # Determine which adapters to touch
            names = list(mod.modules_to_save.keys())
            if adapter_names is not None:
                names = [n for n in names if n in set(adapter_names)]
            if not names:
                continue

            # Inject Adapter
            for an in names:
                A_lin = mod.modules_to_save[an]
                dtype = A_lin.weight.data.dtype
                A = A_lin.weight.data.to(torch.float64) - base_w.data.to(torch.float64)
                A = A.to(dtype)

                # Add LoRA adapters
                peft_mod.add_adapter(an, _lora_conf)

                # Copyt weights to added adapters
                peft_mod.linear.lora_A[an].weight.data.copy_(A)
                peft_mod.linear.lora_B[an].weight.data = torch.eye(
                    out_features, dtype=A.dtype, device=A.device
                )
            # Finally, set mod to final_mod
            # Note: final_mod is not a PEFT model but with LoRA Linear modules inside
            parent, attr = get_parent_and_attr(final_model, mod_name)
            setattr(parent, attr, peft_mod)
            final_model.set_adapter(names[-1])  # set to one of the adapters
        else:
            raise NotImplementedError(f"Module type {type(mod.original_module)} not supported yet.")

    return final_model


def _find_subseq(seq: Sequence[int], subseq: Sequence[int], start_from: int = 0) -> int:
    n, m = len(seq), len(subseq)
    if m == 0:
        return -1
    for i in range(start_from, n - m + 1):
        if list(seq[i : i + m]) == list(subseq):
            return i
    return -1


def _make_labels(input_ids: torch.LongTensor, tokenizer, *, null_id: int = -100):
    """Mask everything except the assistant span.

    We try multiple candidate start/end markers. If none found, we fall back to
    masking up to the last occurrence of the assistant header token ids.
    """
    B, L = input_ids.shape
    labels = torch.full_like(input_ids, fill_value=null_id)

    # Pre-encode candidates once
    start_id_cands = [tokenizer.encode(s, add_special_tokens=False) for s in START_TOKENS]
    end_id_cands = [tokenizer.encode(s, add_special_tokens=False) for s in END_TOKENS]

    for b in range(B):
        seq = input_ids[b].tolist()
        start_idx = -1
        end_idx = -1

        # try all combinations
        for s_ids in start_id_cands:
            s = _find_subseq(seq, s_ids, start_from=0)
            if s != -1:
                # start of answer is just after the header tokens
                start_idx = s + len(s_ids)
                break
        if start_idx == -1:
            raise ValueError(
                "Could not locate assistant start span in tokenized input. "
                "Please customize START_TOKENS for your model's chat template."
            )

        for e_ids in end_id_cands:
            e = _find_subseq(seq, e_ids, start_from=start_idx)
            if e != -1:
                # NOTE: Still unsure whether to include or exclude end marker
                end_idx = e + len(e_ids)
                break
        if end_idx == -1:
            # if no explicit end token, use full sequence tail
            end_idx = L

        if not (0 <= start_idx < end_idx <= L):
            raise ValueError(f"Invalid label span: ({start_idx}, {end_idx}) / L={L}")

        labels[b, start_idx:end_idx] = input_ids[b, start_idx:end_idx]

    return labels


def make_llava_collate(
    processor,
    build_messages_fn,
    build_meta_fn=None,
    with_answer: bool = True,
    with_meta: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    omit_images: bool = False,
    omit_text: bool = False,
):
    tok = processor.tokenizer

    def collate(samples):
        imgs = [s["image"].convert("RGB") for s in samples]
        messages = [
            build_messages_fn(
                s, with_answer=with_answer, omit_images=omit_images, omit_text=omit_text
            )
            for s in samples
        ]

        # 1) text template -> string with <image> token etc. (no tokenize)
        prompts = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not with_answer,
        )

        # ensure no stray trailing newline that can break span search
        prompts = [p[:-1] if len(p) > 0 and p[-1] == "\n" else p for p in prompts]

        # 2) processor does both text + vision
        enc = processor(
            text=prompts,
            images=imgs if not omit_images else None,
            padding=True,
            return_tensors="pt",
        ).to(dtype=dtype)

        # 3) build labels (supervise only assistant span)
        if with_answer:
            # For training, we don't need meta
            enc["labels"] = _make_labels(enc["input_ids"].clone(), tok)
            return enc
        elif with_meta:
            # For evaluation of metrics, we need meta
            assert build_meta_fn is not None, "build_meta_fn must be provided for inference collate"
            meta = build_meta_fn(samples)
            return enc, meta
        else:
            # For unsupervised inference, no meta needed
            return enc

    return collate


def get_collate_fn(
    cfg,
    processor,
    with_answer: bool = True,
    with_meta: bool = False,
    omit_images: bool = False,
    omit_text: bool = False,
):
    build_messages = get_build_messages(cfg)
    build_meta = get_build_meta(cfg)
    return make_llava_collate(
        processor,
        build_messages,
        build_meta_fn=build_meta,
        with_answer=with_answer,
        with_meta=with_meta,
        omit_images=omit_images,
        omit_text=omit_text,
    )


def make_lora_module_filter(target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    def lora_module_filter(module_name, ignore_vision_tower=True, ignore_projector=True):
        # NOTE: This is for Llava only
        if "vision_tower" in module_name and ignore_vision_tower:
            return False
        if "multi_modal_projector" in module_name and ignore_projector:
            return False
        return any(x in module_name for x in target_modules)

    return lora_module_filter


def train_llava(
    cfg,
    train_dataset,
    eval_dataset,
):
    # ------------------------- config -------------------------
    if cfg.logging.use_wandb:
        wandb.init()
    processor = LlavaProcessor.from_pretrained(
        cfg.run.model_id, size=cfg.processor.size, use_fast=cfg.processor.use_fast
    )
    build_messages = get_build_messages(cfg)

    # Load base model
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.run.model_id,
        device_map="auto",
        dtype=torch.bfloat16 if cfg.model.bf16 else torch.float16,
        attn_implementation="flash_attention_2" if cfg.model.flash_attention else "sdpa",
    )

    # Disable cache for training
    model.config.use_cache = not cfg.model.disable_cache_for_training

    # LoRA config (language model modules only; skip vision tower)
    modules_to_save = (
        ["multi_modal_projector.linear_1", "multi_modal_projector.linear_2"]
        if cfg.run.use_projector
        else None
    )
    lora_module_filter = make_lora_module_filter()
    # Expand cfg.lora part
    cfg.lora.target_modules = [
        name for name, _ in model.named_modules() if lora_module_filter(name)
    ]
    cfg.lora.modules_to_save = modules_to_save
    lora_config = get_lora_config(cfg)
    peft_model = get_peft_model(model, lora_config)
    if cfg.run.use_projector:
        for name, param in peft_model.named_parameters():
            if "bias" in name:
                param.requires_grad = False
    peft_model.print_trainable_parameters()

    # --- training args ---
    training_args = SFTConfig(
        output_dir=cfg.run.output_dir,
        report_to="wandb" if cfg.logging.use_wandb else None,
        **dict(cfg.train),
    )

    # We are on training, Thus always with answer
    collate_fn = make_llava_collate(processor, build_messages, with_answer=True)

    trainer_args = dict(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=collate_fn,
    )

    if cfg.run.use_kernel_tracking:
        trainer = KernelSFTTrainer(**trainer_args, use_wandb=cfg.logging.use_wandb)
    else:
        trainer = SFTTrainer(**trainer_args)  # type: ignore

    # Evaluate before training (optional)
    metrics = trainer.evaluate()
    print("***** Eval results before training *****")
    for key, value in metrics.items():
        print(f"  {key} = {value:.4f}")
    # Train and save
    trainer.train()
    peft_model.config.use_cache = True  # type: ignore
    trainer.save_model(training_args.output_dir)


@torch.inference_mode()
def evaluate_llava(cfg, model, test_dataset):
    device = cfg.device if hasattr(cfg, "device") else "auto"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model = get_base_model(cfg)
        if hasattr(cfg, "adapter_save_path") and cfg.adapter_save_path is not None:
            load_lora_model(model, cfg.adapter_save_path)
            model = remove_modules_to_save(model, verbose=True)
            model.set_adapter("default")
            model.merge_and_unload()
        model.to(device)

    collate_fn = get_collate_fn(cfg, model.processor, with_answer=False, with_meta=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return validate_epoch(cfg, model, test_loader, None, model.device)


@torch.inference_mode()
def validate_epoch(cfg, model, test_loader, criterion, device):
    model.eval()
    processor = model.processor
    if processor is None:
        raise ValueError("Processor must be provided in kwargs for validation.")
    process_results = get_process_results(cfg)
    evaluate_results = get_evaluate_results(cfg)

    results = []
    pbar = tqdm(test_loader, desc="Evaluating", ncols=100)
    for i, (enc, meta) in enumerate(pbar):
        # 5) generate (greedy; short answers work best for TextVQA)
        enc = enc.to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        out_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(enc["input_ids"], out)]
        preds = processor.batch_decode(out_trimmed, skip_special_tokens=True)
        results += process_results(meta, preds)

    return evaluate_results(results)
