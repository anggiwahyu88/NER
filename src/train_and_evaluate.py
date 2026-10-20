import json
import random
import time
from pathlib import Path
from collections import Counter
import spacy
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.util import minibatch, compounding

SEED = 42
BASE_MODEL = "xx_ent_wiki_sm"
OUTPUT_DIR = Path("models/trained_ner")
ANNOTATIONS_PATH = Path("data/annotations.jsonl")
EVAL_EXAMPLES_PATH = Path("data/evaluation_examples.json")

HYPERPARAMS = {"n_iter": 20, "dropout": 0.35, "patience": 5}
LABEL_MAP = {
    "PER": "PERSON", "PERSON": "PERSON", "ORG": "ORG",
    "GPE": "GPE", "LOC": "LOC", "DATE": "DATE", "EVENT": "EVENT"
}

def combine_title_body(title: str, body: str) -> str:
    return f"{title.strip()}\n\n{body.strip()}"

def load_and_deduplicate_data(filepath: Path) -> list:
    seen_ids = set()
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item["original_id"] not in seen_ids:
                seen_ids.add(item["original_id"])
                item["text"] = combine_title_body(item["title"], item["body"])
                records.append(item)
    return records

def split_by_article_id(records: list, seed: int = SEED):
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(0.70 * n)
    val_end = train_end + int(0.15 * n)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]

def report_split_statistics(name: str, dataset: list):
    label_counts = Counter()
    for item in dataset:
        for _, _, label in item["entities"]:
            label_counts[label] += 1
    print(f"[{name}] Dokumen: {len(dataset)} | Total Span: {sum(label_counts.values())} | Kelas: {dict(label_counts)}")

def evaluate_model(nlp, dataset: list) -> dict:
    scorer = Scorer()
    examples = []
    for item in dataset:
        pred_doc = nlp(item["text"])
        mapped_ents = []
        for ent in pred_doc.ents:
            if ent.label_ in LABEL_MAP:
                span = pred_doc.char_span(ent.start_char, ent.end_char, label=LABEL_MAP[ent.label_])
                if span is not None:
                    mapped_ents.append(span)
        pred_doc.ents = mapped_ents
        examples.append(Example.from_dict(pred_doc, {"entities": item["entities"]}))

    scores = scorer.score(examples)
    return {
        "precision": round(scores.get("ents_p", 0.0) * 100, 2),
        "recall": round(scores.get("ents_r", 0.0) * 100, 2),
        "f1": round(scores.get("ents_f", 0.0) * 100, 2),
        "per_label": {
            k: {"p": round(v["p"] * 100, 2), "r": round(v["r"] * 100, 2), "f": round(v["f"] * 100, 2)}
            for k, v in (scores.get("ents_per_type") or {}).items()
        }
    }

def analyze_errors(nlp, test_data: list, max_errors: int = 5) -> list:
    errors = []
    for item in test_data:
        doc = nlp(item["text"])
        gold_set = {(s, e, l, item["text"][s:e]) for s, e, l in item["entities"]}
        pred_set = {
            (ent.start_char, ent.end_char, LABEL_MAP.get(ent.label_, ent.label_), ent.text)
            for ent in doc.ents if ent.label_ in LABEL_MAP
        }
        fp, fn = list(pred_set - gold_set), list(gold_set - pred_set)
        if fp or fn:
            errors.append({"original_id": item["original_id"], "false_positives": fp, "false_negatives": fn})
        if len(errors) >= max_errors:
            break
    return errors

def train_pipeline():
    random.seed(SEED)
    spacy.util.fix_random_seed(SEED)

    records = load_and_deduplicate_data(ANNOTATIONS_PATH)
    train_data, val_data, test_data = split_by_article_id(records, seed=SEED)

    report_split_statistics("TRAIN", train_data)
    report_split_statistics("VAL", val_data)
    report_split_statistics("TEST", test_data)

    nlp = spacy.load(BASE_MODEL)
    print("Baseline Val:", evaluate_model(nlp, val_data))
    print("Baseline Test:", evaluate_model(nlp, test_data))

    ner = nlp.get_pipe("ner") if "ner" in nlp.pipe_names else nlp.add_pipe("ner", last=True)
    for item in train_data:
        for _, _, label in item["entities"]:
            ner.add_label(label)

    train_examples = [
        Example.from_dict(nlp.make_doc(item["text"]), {"entities": item["entities"]})
        for item in train_data
    ]

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    best_val_f1, best_model_bytes, patience_counter = -1.0, None, 0
    start_time = time.time()

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        for epoch in range(1, HYPERPARAMS["n_iter"] + 1):
            random.shuffle(train_examples)
            losses = {}
            for batch in minibatch(train_examples, size=compounding(4.0, 16.0, 1.001)):
                nlp.update(batch, drop=HYPERPARAMS["dropout"], losses=losses, sgd=optimizer)

            val_f1 = evaluate_model(nlp, val_data)["f1"]
            print(f"Epoch {epoch:02d} | Loss: {losses.get('ner', 0.0):.4f} | Val F1: {val_f1}%")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_bytes = nlp.to_bytes()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= HYPERPARAMS["patience"]:
                    break

    print(f"Durasi Pelatihan: {round(time.time() - start_time, 2)} detik")
    if best_model_bytes:
        nlp.from_bytes(best_model_bytes)

    print("Trained Val:", evaluate_model(nlp, val_data))
    print("Trained Test:", evaluate_model(nlp, test_data))
    print("Contoh Error Test Set:", json.dumps(analyze_errors(nlp, test_data)[:3], indent=2, ensure_ascii=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(OUTPUT_DIR)

    EVAL_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    train_pipeline()