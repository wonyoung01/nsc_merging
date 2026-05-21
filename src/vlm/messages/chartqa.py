def build_messages(s, with_answer, omit_images=False, omit_text=False):
    """
    Build the message structure for LLaVA-style chat. (Based on liuhaotian/LLaVA code)
    If with_answer is False, the answer field is ignored and not included in the messages.
    """
    if omit_images and omit_text:
        raise ValueError("Both omit_images and omit_text cannot be True.")
    # Unpack the sample
    img, question, answer = s["image"].convert("RGB"), s["query"], s["label"][0]
    # Set default message
    post_prompt = "\nAnswer the question with a single word."
    # one-image, single-turn chat (LLaVA supports the same multimodal messages schema)
    content = []
    if not omit_images:
        content.append({"type": "image", "image": img})
    if not omit_text:
        content.append({"type": "text", "text": question + post_prompt})
    return [
        {
            "role": "user",
            "content": content,
        },
    ] + (
        [{"role": "assistant", "content": [{"type": "text", "text": answer + "</s>"}]}]
        if with_answer
        else []
    )


def build_meta(samples):
    qs = [s["query"] for s in samples]
    gts = [s["label"] for s in samples]
    meta = {
        "questions": qs,
        "answers": gts,
    }

    return meta


def process_results(meta, preds):
    gts = meta["answers"]
    return [{"answer": answer[0], "pred": pred} for answer, pred in zip(gts, preds)]


def evaluate_results(results):
    gts = [r["answer"] for r in results]
    preds = [r["pred"] for r in results]
    correct = 0
    for gt, pred in zip(gts, preds):
        if relaxed_correctness(pred, gt):
            correct += 1
    accuracy = correct / len(gts) if len(gts) > 0 else 0.0
    return {"relaxed_accuracy": accuracy}


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
