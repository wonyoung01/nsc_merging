def build_messages(s, with_answer, omit_images=False, omit_text=False):
    if omit_images and omit_text:
        raise ValueError("Both omit_images and omit_text cannot be True.")
    # NOTE: Using only first answer for the labeling
    img, question, answer = s["image"].convert("RGB"), s["question"], s["answers"][0]

    # one-image, single-turn chat
    post_prompt = "\nAnswer the question with a single word."
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
    qs = [s["question"] for s in samples]
    gts = [s["answers"] for s in samples]
    meta = {
        "questions": qs,
        "answers": gts,
    }
    return meta


def process_results(meta, preds):
    gts = meta["answers"]
    return [{"answer": answer, "pred": pred} for answer, pred in zip(gts, preds)]


def evaluate_results(results):
    gts = [r["answer"] for r in results]
    preds = [r["pred"] for r in results]
    anls_scores = anls_correctness(preds, gts)
    average_anls = sum(anls_scores) / len(anls_scores) if len(anls_scores) > 0 else 0.0
    return {"ANLS": average_anls}


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls(references, prediction, thresh_hold=0.5):
    """https://github.com/QwenLM/Qwen-VL/blob/master/eval_mm/infographicsvqa_eval.py"""
    values = []
    # Unwrap predictions if it's a nested list
    pred = prediction[0] if isinstance(prediction, list) else prediction

    for answer in references:
        # preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < thresh_hold:
        question_result = 0
    return question_result


def anls_correctness(prediction, target):
    assert len(prediction) == len(target), "Length mismatch"
    hits = []
    for p, t in zip(prediction, target):
        hits.append(anls(t, p))
    return hits
