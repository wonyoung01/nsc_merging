import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor


class LlavaWrapper(nn.Module):
    """
    A thin proxy so that `wrapper.vit` == inner LLaVA cond-gen model,
    while preserving all HF/PEFT behaviors.
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        size: int = 336,
        use_fast: bool = True,
    ):
        super().__init__()
        # Register the underlying model as a submodule.
        self.base_id = model_id
        self.vit: LlavaForConditionalGeneration = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )
        self.processor = LlavaProcessor.from_pretrained(
            model_id,
            size=size,
            use_fast=use_fast,
        )

    # ---- Interface you want to expose ----
    def forward(self, *args, **kwargs):
        # Forward to the wrapped model
        return self.vit(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.vit.generate(*args, **kwargs)

    # Optional: mirror frequently used HF helpers
    def save_pretrained(self, *args, **kwargs):
        return self.vit.save_pretrained(*args, **kwargs)

    def push_to_hub(self, *args, **kwargs):
        return self.vit.push_to_hub(*args, **kwargs)

    def differentiable_decode(self, data, **decode_kwargs):
        # A simple wrapper around the differentiable decoding function
        return _differentiable_decode(self, data, **decode_kwargs)

    # ---- Delegation fallback ----
    def __getattr__(self, name):
        # nn.Module uses __getattr__ for submodule/param lookup,
        # so first try super(); on failure, delegate to inner model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vit, name)


def _differentiable_decode(
    model,
    data,
    use_cache=False,
    max_new_tokens=128,
    temperature=None,
    top_k=None,
    do_sample=False,
    stop_token_id=None,  # or tokenizer.eos_token_id
    report_entropies=False,
    report_logprobs=False,
    report_decoded=False,
    **kwargs,
):
    """
    Returns:
    tokens: LongTensor [1, total_len] (prompt + generated)
    entropies: list of scalar tensors for each step (requires_grad=True)
    step_logprobs: list of scalar tensors log p(token_t | prefix) (requires_grad=True)
    """
    processor = model.processor
    tokenizer = processor.tokenizer
    if stop_token_id is None:
        stop_token_id = tokenizer.eos_token_id
    loss_cluster = kwargs.get("loss_cluster", None)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    pixel_values = data.get("pixel_values", None)

    # past_key_values speeds up decoding; keep grads for current step only (standard practice)
    past_key_values = None

    entropies = []
    step_logprobs = []
    decoded = ""
    generated = input_ids

    for step in range(max_new_tokens):
        outputs = model(
            input_ids=generated if past_key_values is None else generated[:, -1:],
            attention_mask=attention_mask if past_key_values is None else None,
            pixel_values=pixel_values,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        if use_cache:
            past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]  # [1, vocab]
        if temperature and temperature != 1.0:
            next_logits = next_logits / temperature

        # optional top-k
        if top_k is not None and top_k > 0:
            top_values, top_indices = torch.topk(next_logits, k=top_k, dim=-1)
            mask = torch.full_like(next_logits, float("-inf"))
            next_logits = mask.scatter(1, top_indices, top_values)

        next_logprobs = F.log_softmax(next_logits, dim=-1)  # grads flow through logits
        next_probs = next_logprobs.exp()

        # entropy H(p) = -sum p log p (grads wrt logits via softmax/logsoftmax)
        H = -(next_probs * next_logprobs).sum(dim=-1).mean()
        if report_entropies:
            entropies.append(H.item())
        if kwargs.get("backprop_entropy", False):
            balance = kwargs.get("entropy_balance", 1.0)
            loss = H / balance
            loss.backward()

        # choose next token (non-differentiable choice; that’s fine)
        if do_sample:
            with torch.no_grad():
                next_token = torch.multinomial(next_probs, num_samples=1)  # [1,1]
        else:
            with torch.no_grad():
                next_token = torch.argmax(next_logprobs, dim=-1, keepdim=True)  # [1,1]

        # logprob of the chosen token (grad wrt logits, token id detached)
        lp = next_logprobs.gather(1, next_token).squeeze(1).mean()
        if report_logprobs:
            step_logprobs.append(lp)

        # append token & update masks
        generated = torch.cat([generated, next_token], dim=1)
        if attention_mask is not None:
            pad = torch.ones_like(next_token)
            attention_mask = torch.cat([attention_mask, pad], dim=1)

        # early stop
        if step + 1 >= max_new_tokens:
            break
        if stop_token_id is not None and int(next_token.item()) == int(stop_token_id):
            break
        if loss_cluster is not None:
            loss_cluster.clear()

    if report_decoded:
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    return (generated, entropies, step_logprobs, decoded)


def _test1():
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    model = LlavaWrapper()
    processor = AutoProcessor.from_pretrained(model.base_id)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are these?"},
                {"type": "image", "image": raw_image},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    data = processor(images=raw_image, text=prompt, padding=True, return_tensors="pt").to(
        0, torch.bfloat16
    )
    model.to(0)

    output = model.generate(**data, max_new_tokens=200, do_sample=False)
    print(processor.decode(output[0], skip_special_tokens=True))


def _test2():
    # Example usage (sketch):
    # model: LlavaForConditionalGeneration; tokenizer & image_processor matched to the checkpoint
    # prompt = "USER: <image>\nDescribe the image concisely.\nASSISTANT:"
    # image = PIL.Image.open("example.jpg")
    # tokens, entropies, logps = differentiable_decode(model, tok, img_proc, prompt, image, do_sample=True, entropy_coef=0.01)
    # # Build a loss, e.g., REINFORCE-style surrogate or entropy-regularized NLL:
    # # Here’s a simple entropy-regularized imitation loss for illustration:
    # loss = 0.0
    # for H, lp in zip(entropies, logps):
    #     loss = loss - lp - 0.01 * H
    # loss.backward()
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    model = LlavaWrapper()
    processor = model.processor
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are these?"},
                {"type": "image", "image": raw_image},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    data = processor(images=raw_image, text=prompt, padding=True, return_tensors="pt").to(
        0, torch.bfloat16
    )
    # Make sure multimodal projecton layers are the only trainable params
    for name, param in model.named_parameters():
        if "multimodal_projection" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.to(0)
    with torch.enable_grad():
        tokens, entropies, logps, decoded = _differentiable_decode(
            model,
            data,
            use_cache=False,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_k=None,
            stop_token_id=processor.tokenizer.eos_token_id,
            report_entropies=True,
            report_logprobs=True,
            report_decoded=True,
        )
    print("Decoded output:", decoded)


if __name__ == "__main__":
    option = 2
    if option == 1:
        _test1()
    elif option == 2:
        _test2()
    else:
        raise ValueError("Invalid option")
