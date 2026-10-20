import json
import re
from pathlib import Path
import spacy
from load_and_profile import load_and_profile_dataset, SAMPLE_DIR

OUTPUT_JSONL = Path("data/annotations.jsonl")
BASE_MODEL = "xx_ent_wiki_sm"  # Mendukung multi-bahasa (en, ms/id, zh)

KNOWN_ORGS = [
    "Maybank Malaysia", "Maybank Group", "Maybank Foundation", "Maybank CSR", "Maybank",
    "Hong Leong Bank", "CIMB Group Malaysia", "CIMB Group", "CIMB",
    "Public Bank Berhad", "Public Bank", "RHB Bank", "RHB", "MDV", "大众银行"
]

LABEL_MAP = {
    "PER": "PERSON", "PERSON": "PERSON",
    "ORG": "ORG", "GPE": "GPE", "LOC": "LOC",
    "DATE": "DATE", "EVENT": "EVENT"
}

def find_dictionary_spans(text: str, terms: list, label: str) -> list:
    spans = []
    for term in sorted(terms, key=len, reverse=True):
        if not term:
            continue
        for match in re.finditer(re.escape(term), text):
            spans.append((match.start(), match.end(), label))
    return spans

def resolve_overlapping_spans(spans: list) -> list:
    sorted_spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    filtered = []
    last_end = -1
    for start, end, label in sorted_spans:
        if start >= last_end:
            filtered.append([start, end, label])
            last_end = end
    return filtered

def generate_initial_annotations(max_articles: int = 80):
    raw_df, _ = load_and_profile_dataset(SAMPLE_DIR)
    valid_df = raw_df[(raw_df["title"] != "") & (raw_df["body"] != "")].copy()
    valid_df = valid_df.drop_duplicates(subset=["original_id"], keep="first")
    subset_df = valid_df.head(max_articles)

    nlp = spacy.load(BASE_MODEL)
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in subset_df.iterrows():
            title, body = row["title"], row["body"]
            combined_text = f"{title}\n\n{body}"
            candidate_spans = []

            doc = nlp(combined_text)
            for ent in doc.ents:
                if ent.label_ in LABEL_MAP:
                    candidate_spans.append((ent.start_char, ent.end_char, LABEL_MAP[ent.label_]))

            candidate_spans.extend(find_dictionary_spans(combined_text, KNOWN_ORGS, "ORG"))
            spokespersons = [row.get("spokePerson1", ""), row.get("spokePerson2", ""), row.get("spokePerson3", "")]
            candidate_spans.extend(find_dictionary_spans(combined_text, spokespersons, "PERSON"))

            record = {
                "original_id": row["original_id"],
                "source_csv": row["source_csv_file"],
                "source_row": int(row["source_row_index"]),
                "language": row["language"],
                "title": title,
                "body": body,
                "entities": resolve_overlapping_spans(candidate_spans)
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    generate_initial_annotations()