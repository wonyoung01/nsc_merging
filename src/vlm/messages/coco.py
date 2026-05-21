from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

FLICKR_METRICS = [
    "Bleu_4",
    "Bleu_3",
    "Bleu_2",
    "Bleu_1",
    "METEOR",
    "ROUGE_L",
    "CIDEr",
]  # , "SPICE"]


def build_messages(s, with_answer, omit_images=False, omit_text=False):
    if omit_images and omit_text:
        raise ValueError("Both omit_images and omit_text cannot be True.")
    # Unpack the sample
    img = s["image"].convert("RGB")
    sentences = s["sentences"]
    if "raw" in sentences:
        answer = sentences["raw"].strip()
    else:
        answer = sentences[0]["raw"].strip()

    # one-image, single-turn chat
    post_prompt = "Provide a one-sentence caption for the provided image."
    content = []
    if not omit_images:
        content.append({"type": "image", "image": img})
    if not omit_text:
        content.append({"type": "text", "text": post_prompt})
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
    gts = []
    for s in samples:
        if "raw" in s["sentences"]:
            gts.append(s["sentences"]["raw"].strip())
        else:
            gts.append([a["raw"].strip() for a in s["sentences"]])
    img_id = [s["imgid"] for s in samples]
    cocoid = [s["cocoid"] for s in samples]
    meta = {
        "answer": gts,
        "image_id": cocoid,
        "id": img_id,
    }
    return meta


def process_results(meta, preds):
    gts = meta["answer"]
    img_ids = meta["image_id"]
    ids = meta["id"]
    img_ids = [int(i) for i in img_ids]
    return [
        {"answer": answer, "pred": pred, "image_id": img_id, "id": id}
        for answer, pred, img_id, id in zip(gts, preds, img_ids, ids)
    ]


def evaluate_results(results):
    # NOTE: only compute CIDEr for now to save time
    cider_score = coco_cider(results)
    return {"CIDEr": cider_score}


def coco_aggregation_result(results, metric):
    scorers = [
        (Bleu(4), "Bleu_1"),
        (Bleu(4), "Bleu_2"),
        (Bleu(4), "Bleu_3"),
        (Bleu(4), "Bleu_4"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]  # , (Spice(), "SPICE")]
    scorers_dict = {s[1]: s for s in scorers}
    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    for result in results:
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})
        for a in result["answer"]:
            dataset["annotations"].append(
                {"image_id": int(result["image_id"]), "caption": a, "id": idx}
            )
            idx += 1
        dataset["images"].append({"id": int(result["image_id"])})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)

    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    return score


def coco_bleu4(results):
    return coco_aggregation_result(results, "Bleu_4")


def coco_bleu3(results):
    return coco_aggregation_result(results, "Bleu_3")


def coco_bleu2(results):
    return coco_aggregation_result(results, "Bleu_2")


def coco_bleu1(results):
    return coco_aggregation_result(results, "Bleu_1")


def coco_meteor(results):
    return coco_aggregation_result(results, "METEOR")


def coco_rougel(results):
    return coco_aggregation_result(results, "ROUGE_L")


def coco_cider(results):
    return coco_aggregation_result(results, "CIDEr")


def coco_spice(results):
    return coco_aggregation_result(results, "SPICE")
