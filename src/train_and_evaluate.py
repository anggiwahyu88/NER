import json
import random
import time
from pathlib import Path
from collections import Counter
import spacy
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.util import minibatch

SEED = 42
BASE_MODEL = "xx_ent_wiki_sm"
OUTPUT_DIR = Path("models/trained_ner")
ANNOTATIONS_PATH = Path("data/annotations.jsonl")
EVAL_EXAMPLES_PATH = Path("data/evaluation_examples.json")

HYPERPARAMS = {"n_iter": 15, "dropout": 0.0, "patience": 5}

KNOWN_ORGS = [
    "Maybank Malaysia", "Maybank Group", "Maybank Foundation", "Maybank CSR", "Maybank",
    "Malayan Banking Bhd", "Malayan Banking",
    "Hong Leong Bank", "CIMB Group Malaysia", "CIMB Group", "CIMB",
    "Public Bank Berhad", "Public Bank", "RHB Bank", "RHB", "MDV", "大众银行"
]

MODEL_TO_EVAL_LABEL = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "LOC",
    "DATE": "DATE",
    "EVENT": "EVENT"
}

EVAL_TO_MODEL_LABEL = {
    "PERSON": "PER",
    "ORG": "ORG",
    "LOC": "LOC",
    "GPE": "GPE",
    "DATE": "DATE",
    "EVENT": "EVENT"
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


def get_aligned_entities(doc, entities: list, to_model_label: bool = False) -> list:
    aligned = []
    used_tokens = set()
    for start, end, label in entities:
        target_label = EVAL_TO_MODEL_LABEL.get(label, label) if to_model_label else label
        span = doc.char_span(start, end, label=target_label, alignment_mode="strict")
        if span is None:
            span = doc.char_span(start, end, label=target_label, alignment_mode="contract")
        if span is None or len(span.text.strip()) == 0:
            continue
        token_indices = set(range(span.start, span.end))
        if not (token_indices & used_tokens):
            used_tokens.update(token_indices)
            aligned.append([span.start_char, span.end_char, target_label])
    return aligned


def build_rehearsal_training_examples(base_nlp, train_data: list) -> list:
    """
    Membuat Example untuk training sambil mempertahankan prediksi MISC bawaan model
    pada token yang tidak berlabel (rehearsal) agar tidak terjadi catastrophic forgetting.
    """
    examples = []
    for item in train_data:
        base_doc = base_nlp(item["text"])
        gold_spans = get_aligned_entities(base_doc, item["entities"], to_model_label=True)

        used_tokens = set()
        combined_entities = []

        for s, e, label in gold_spans:
            span = base_doc.char_span(s, e, label=label)
            if span is not None:
                used_tokens.update(range(span.start, span.end))
                combined_entities.append([s, e, label])

        # Pertahankan entitas MISC dari model dasar jika tidak bertabrakan dengan gold_spans
        for ent in base_doc.ents:
            if ent.label_ == "MISC":
                ent_tokens = set(range(ent.start, ent.end))
                if not (ent_tokens & used_tokens):
                    used_tokens.update(ent_tokens)
                    combined_entities.append([ent.start_char, ent.end_char, "MISC"])

        raw_doc = base_nlp.make_doc(item["text"])
        examples.append(Example.from_dict(raw_doc, {"entities": combined_entities}))
    return examples


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
            if ent.label_ in MODEL_TO_EVAL_LABEL:
                eval_label = MODEL_TO_EVAL_LABEL[ent.label_]
                span = pred_doc.char_span(ent.start_char, ent.end_char, label=eval_label)
                if span is not None:
                    mapped_ents.append(span)
        pred_doc.ents = mapped_ents

        clean_gold = get_aligned_entities(pred_doc, item["entities"], to_model_label=False)
        examples.append(Example.from_dict(pred_doc, {"entities": clean_gold}))

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
        clean_gold = get_aligned_entities(doc, item["entities"], to_model_label=False)
        gold_set = {(s, e, l, item["text"][s:e]) for s, e, l in clean_gold}
        pred_set = {
            (ent.start_char, ent.end_char, MODEL_TO_EVAL_LABEL.get(ent.label_, ent.label_), ent.text)
            for ent in doc.ents if ent.label_ in MODEL_TO_EVAL_LABEL
        }
        fp = [list(x) for x in sorted(pred_set - gold_set)]
        fn = [list(x) for x in sorted(gold_set - pred_set)]
        if fp or fn:
            errors.append({
                "original_id": item["original_id"],
                "false_positives": fp,
                "false_negatives": fn
            })
        if len(errors) >= max_errors:
            break
    return errors


def train_pipeline():
    random.seed(SEED)
    spacy.util.fix_random_seed(SEED)

    records = load_and_deduplicate_data(ANNOTATIONS_PATH)
    train_data, val_data, test_data = split_by_article_id(records, seed=SEED)

    print("=== 1. STATISTIK DATA SPLIT (UNTUK SLIDE 5) ===")
    report_split_statistics("TRAIN (70%)", train_data)
    report_split_statistics("VAL   (15%)", val_data)
    report_split_statistics("TEST  (15%)", test_data)

    nlp = spacy.load(BASE_MODEL)
    print("\n=== 2. PERFORMA MODEL BASELINE (SEBELUM FINE-TUNING) ===")
    baseline_val = evaluate_model(nlp, val_data)
    baseline_test = evaluate_model(nlp, test_data)
    print("Baseline Validation :", json.dumps(baseline_val))
    print("Baseline Test       :", json.dumps(baseline_test))

    # Bangun contoh training dengan pseudo-rehearsal untuk label MISC
    train_examples = build_rehearsal_training_examples(nlp, train_data)

    ner = nlp.get_pipe("ner")
    for ex in train_examples:
        for ent in ex.reference.ents:
            ner.add_label(ent.label_)

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    best_val_f1 = baseline_val["f1"]
    best_model_bytes = nlp.to_bytes()
    patience_counter = 0
    start_time = time.time()

    print("\n=== PROSES FINE-TUNING ===")
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        optimizer.learn_rate = 0.00001
        optimizer.L2 = 0.0
        optimizer.grad_clip = 1.0

        for epoch in range(1, HYPERPARAMS["n_iter"] + 1):
            random.shuffle(train_examples)
            losses = {}
            for batch in minibatch(train_examples, size=16):
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
                    print(f"Early stopping dihentikan pada epoch {epoch}.")
                    break

    print(f"Durasi Pelatihan: {round(time.time() - start_time, 2)} detik")
    if best_model_bytes:
        nlp.from_bytes(best_model_bytes)

    # Tambahkan EntityRuler setelah neural NER untuk memastikan entitas domain terdeteksi konsisten
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
        ruler.add_patterns([{"label": "ORG", "pattern": org} for org in KNOWN_ORGS])

    print("\n=== 3. PERFORMA MODEL TRAINED (SESUDAH FINE-TUNING - UNTUK SLIDE 6) ===")
    print("Trained Validation  :", json.dumps(evaluate_model(nlp, val_data)))
    print("Trained Test        :", json.dumps(evaluate_model(nlp, test_data)))

    print("\n=== 4. ANALISIS ERROR TEST SET (UNTUK SLIDE 7) ===")
    print(json.dumps(analyze_errors(nlp, test_data, max_errors=3), indent=2, ensure_ascii=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(OUTPUT_DIR)

    EVAL_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    train_pipeline()