def build_messages(s, with_answer, omit_images=False, omit_text=False):
    if omit_images and omit_text:
        raise ValueError("Both omit_images and omit_text cannot be True.")
    img, question, answer = (
        s["image"].convert("RGB"),
        s["question"],
        s["answer"],
    )

    content = []
    if not omit_images:
        content.append({"type": "image", "image": img})
    if not omit_text:
        content.append({"type": "text", "text": question})
    # Full question prompt
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
    gts = [s["answer"] for s in samples]
    meta = {
        "questions": qs,
        "answers": gts,
    }
    return meta


def process_results(meta, preds):
    gts = meta["answers"]
    return [{"answer": answer, "pred": pred} for answer, pred in zip(gts, preds)]


def evaluate_results(results):
    targets = [res["answer"] for res in results]
    preds = [res["pred"] for res in results]
    hits = iconqa_hits(targets, preds)
    total = len(targets)
    accuracy = hits / total if total > 0 else 0.0
    return {"accuracy": accuracy}


def iconqa_hits(targets, results):
    targets = [target.strip().lower() for target in targets]
    hits = 0
    for t, r in zip(targets, results):
        if r.lower() == t:
            hits += 1
        elif len(r) >= 2 and r[0].isupper() and r[1] == ".":
            if r[0].lower() == t:
                hits += 1
    return hits
