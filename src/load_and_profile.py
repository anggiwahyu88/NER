import json
from pathlib import Path
import pandas as pd

SAMPLE_DIR = Path("../use_cases/pnm_study_case/sample")
IMPORTANT_COLS = [
    "original_id", "source_name", "title", "body",
    "language", "entities", "date_published", "clipping",
    "spokePerson1", "spokePerson2", "spokePerson3"
]


def clean_outer_quotes(val) -> str:
    """Membersihkan tanda kutip pembungkus di ujung string tanpa merusak kutipan di dalam kalimat."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
        return s[1:-1].strip()
    if s.startswith("'"):
        return s[1:].strip()
    return s


def load_and_profile_dataset(sample_dir: Path):
    csv_files = sorted(sample_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file CSV di {sample_dir}")

    all_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(
            csv_file,
            sep=";",
            dtype=str,
            quotechar='"',
            on_bad_lines="warn",
            encoding="utf-8"
        )
        df["source_csv_file"] = csv_file.name
        df["source_row_index"] = df.index
        all_dfs.append(df)

    raw_df = pd.concat(all_dfs, ignore_index=True)

    for col in IMPORTANT_COLS:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].apply(clean_outer_quotes)

    report = {
        "files_loaded": len(csv_files),
        "total_rows_loaded": len(raw_df),
        "duplicate_rows_by_original_id": int(raw_df.duplicated(subset=["original_id"]).sum()),
        "duplicate_rows_by_title_body": int(raw_df.duplicated(subset=["title", "body"]).sum()),
        "missing_titles": int((raw_df["title"] == "").sum()),
        "missing_bodies": int((raw_df["body"] == "").sum()),
        "language_distribution": raw_df["language"].replace("", "UNKNOWN").value_counts().to_dict(),
        "rows_with_non_empty_entities_col": int((raw_df["entities"] != "").sum()),
    }

    print("=== LAPORAN PROFILING DATASET ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return raw_df, report


if __name__ == "__main__":
    load_and_profile_dataset(SAMPLE_DIR)