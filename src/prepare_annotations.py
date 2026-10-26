import json
import re
from pathlib import Path
import spacy
from load_and_profile import load_and_profile_dataset, SAMPLE_DIR

OUTPUT_JSONL = Path("data/annotations.jsonl")
BASE_MODEL = "xx_ent_wiki_sm"  # Mendukung multi-bahasa (en, ms/id, zh)

KNOWN_ORGS = [
    "Maybank Malaysia", "Maybank Group", "Maybank Foundation", "Maybank CSR", "Maybank",
    "Malayan Banking Bhd", "Malayan Banking",
    "Hong Leong Bank", "CIMB Group Malaysia", "CIMB Group", "CIMB",
    "Public Bank Berhad", "Public Bank", "RHB Bank", "RHB", "MDV", "大众银行"
]

LABEL_MAP = {
    "PER": "PERSON", "PERSON": "PERSON",
    "ORG": "ORG", "GPE": "GPE", "LOC": "LOC",
    "DATE": "DATE", "EVENT": "EVENT"
}


def find_dictionary_spans(text: str, terms: list, label: str, priority: int = 1) -> list:
    """Mencari kecocokan kamus dengan word boundary untuk teks ASCII."""
    spans = []
    for term in sorted(terms, key=len, reverse=True):
        term = str(term).strip()
        if not term:
            continue
        if term.isascii():
            pattern = rf"\b{re.escape(term)}\b"
        else:
            pattern = re.escape(term)
        for match in re.finditer(pattern, text):
            spans.append((match.start(), match.end(), label, priority))
    return spans


def resolve_and_align_spans(doc, candidate_spans: list) -> list:
    """
    Menyelesaikan tumpang-tindih span berdasarkan prioritas & panjang span,
    serta menyelaraskan batas karakter dengan batas token spaCy (mencegah W030).
    Urutan sort: (prioritas lebih tinggi/angka kecil, span lebih panjang, posisi awal).
    """
    # Urutkan berdasarkan: prioritas (1 duluan), panjang span (terpanjang duluan), start index
    sorted_candidates = sorted(
        candidate_spans,
        key=lambda x: (x[3], -(x[1] - x[0]), x[0])
    )

    used_tokens = set()
    aligned_spans = []

    for start, end, label, _ in sorted_candidates:
        # Coba selaraskan secara ketat terlebih dahulu, lalu contract jika ada spasi/tanda baca
        span = doc.char_span(start, end, label=label, alignment_mode="strict")
        if span is None:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            continue

        span_token_indices = set(range(span.start, span.end))
        if not (span_token_indices & used_tokens):
            used_tokens.update(span_token_indices)
            aligned_spans.append([span.start_char, span.end_char, label])

    # Urutkan kembali berdasarkan posisi kemunculan di teks
    aligned_spans.sort(key=lambda x: x[0])
    return aligned_spans


def generate_initial_annotations(max_articles: int = 80):
    raw_df, _ = load_and_profile_dataset(SAMPLE_DIR)
    valid_df = raw_df[(raw_df["title"] != "") & (raw_df["body"] != "")].copy()
    valid_df = valid_df.drop_duplicates(subset=["original_id"], keep="first")
    subset_df = valid_df.head(max_articles)

    nlp = spacy.load(BASE_MODEL)
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in subset_df.iterrows():
            title, body = row["title"].strip(), row["body"].strip()
            combined_text = f"{title}\n\n{body}"
            doc = nlp(combined_text)

            candidate_spans = []

            # Prioritas 1: Kamus Organisasi & Spokesperson (paling tepercaya)
            candidate_spans.extend(find_dictionary_spans(combined_text, KNOWN_ORGS, "ORG", priority=1))
            spokespersons = [
                row.get("spokePerson1", ""),
                row.get("spokePerson2", ""),
                row.get("spokePerson3", "")
            ]
            candidate_spans.extend(find_dictionary_spans(combined_text, spokespersons, "PERSON", priority=1))

            # Prioritas 2: Prediksi bawaan spaCy
            for ent in doc.ents:
                if ent.label_ in LABEL_MAP:
                    candidate_spans.append((
                        ent.start_char,
                        ent.end_char,
                        LABEL_MAP[ent.label_],
                        2
                    ))

            clean_entities = resolve_and_align_spans(doc, candidate_spans)

            record = {
                "original_id": row["original_id"],
                "source_csv": row["source_csv_file"],
                "source_row": int(row["source_row_index"]),
                "language": row["language"],
                "title": title,
                "body": body,
                "entities": clean_entities
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_written += 1

    print(f"Berhasil menyimpan {total_written} dokumen anotasi ke {OUTPUT_JSONL}")


if __name__ == "__main__":
    generate_initial_annotations()