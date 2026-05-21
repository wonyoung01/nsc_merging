from .m4c_evaluator import TextVQAAccuracyEvaluator

EVALUATOR = TextVQAAccuracyEvaluator()


def build_messages(
    s,
    with_answer,
    omit_images=False,
    omit_text=False,
):
    if omit_images and omit_text:
        raise ValueError("Both omit_images and omit_text cannot be True.")
    question, answer, img = s["question"], s["answers"][0], s["image"].convert("RGB")
    # Full question prompt
    post_prompt = (
        "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
        + "\nAnswer the question using a single word or phrase."
    )
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
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer + "</s>"}],
            }
        ]
        if with_answer
        else []
    )


def build_meta(samples):
    qs = [s["question"] for s in samples]
    qs_ids = [s["filename"] for s in samples]
    gts = [s["answers"] for s in samples]
    meta = {
        "questions": qs,
        "question_id": qs_ids,
        "answers": gts,
    }
    return meta


def process_results(meta, preds):
    gts = meta["answers"]
    return [{"gt_answers": answer, "pred_answer": pred} for answer, pred in zip(gts, preds)]


def evaluate_results(results):
    exact_match = EVALUATOR.eval_pred_list(results)
    return {"exact_match": exact_match}
