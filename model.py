from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_NAME = "unitary/toxic-bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

pipeline = TextClassificationPipeline(
    model = model,
    tokenizer = tokenizer,
    return_all_scores = True,
)

LABEL_MAPPING = {
    0:"toxic",
    1:"severe_toxic",
    2:"obscene",
    3:"threat",
    4:"insult",
    5:"identity_hate",
}

def analyze_text(txt: str):
    outputs = pipeline(txt)[0]
    max_label = max(outputs, key= lambda x: x["score"])
    overall_score = max_label["score"]
    label_ids = max_label["label"]

    if "_" in label_ids:
        idx = int(label_ids.split("_")[-1])
        label = LABEL_MAPPING.get(idx, "toxic")
    else:
        label = label_ids


    all_scores = {out["label"]: float(out["score"]) for out in outputs}

    return {
        "primary_label": label,
        "primary_score" : overall_score,
        "all_scores" : all_scores,
    }