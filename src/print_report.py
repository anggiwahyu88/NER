import json
import spacy
from train_and_evaluate import (
    ANNOTATIONS_PATH, BASE_MODEL, OUTPUT_DIR, SEED,
    load_and_deduplicate_data, split_by_article_id,
    report_split_statistics, evaluate_model, analyze_errors
)

def generate_slide_report():
    records = load_and_deduplicate_data(ANNOTATIONS_PATH)
    train_data, val_data, test_data = split_by_article_id(records, seed=SEED)

    print("=== 1. STATISTIK DATA SPLIT (UNTUK SLIDE 5) ===")
    report_split_statistics("TRAIN (70%)", train_data)
    report_split_statistics("VAL   (15%)", val_data)
    report_split_statistics("TEST  (15%)", test_data)

    print("\n=== 2. PERFORMA MODEL BASELINE (SEBELUM FINE-TUNING) ===")
    nlp_base = spacy.load(BASE_MODEL)
    print("Baseline Validation :", json.dumps(evaluate_model(nlp_base, val_data)))
    print("Baseline Test       :", json.dumps(evaluate_model(nlp_base, test_data)))

    print("\n=== 3. PERFORMA MODEL TRAINED (SESUDAH FINE-TUNING - UNTUK SLIDE 6) ===")
    nlp_trained = spacy.load(OUTPUT_DIR)
    print("Trained Validation  :", json.dumps(evaluate_model(nlp_trained, val_data)))
    print("Trained Test        :", json.dumps(evaluate_model(nlp_trained, test_data)))

    print("\n=== 4. ANALISIS ERROR TEST SET (UNTUK SLIDE 7) ===")
    errors = analyze_errors(nlp_trained, test_data, max_errors=3)
    print(json.dumps(errors, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    generate_slide_report()