"""
Scores each model's extraction against your hand-labeled ground truth.

Scoring rules (documented here because this IS the methodology the
screening task is grading):

- vendor:        fuzzy match (difflib similarity ratio), threshold 0.80
- gst_details:    fuzzy match, threshold 0.60 (free text, often sparse/absent)
- invoice_number: normalized exact match (lowercase, whitespace/punctuation stripped)
- currency:       normalized exact match (uppercase, stripped)
- date:           parsed with dateutil (dayfirst=True for Indian date formats),
                   compared as actual date objects. Falls back to normalized
                   string match if parsing fails on either side.
- amount:         parsed as float (currency symbols/commas stripped), compared
                   after rounding to 2 decimal places.

For every field: if ground truth is null/empty AND prediction is null/empty,
that's counted as CORRECT (the model correctly recognized nothing was there).
If ground truth has a value but the prediction is null, or vice versa, that's
INCORRECT.

Cost is computed from each result's recorded token usage x published
per-1M-token pricing (see PRICING dict below - update if provider prices
change).

Usage:
    python compare.py
"""

import os
import re
import json
import glob
import difflib
from datetime import datetime

import pandas as pd
from dateutil import parser as dateparser

OUTPUT_DIR = "output"
TABLES_DIR = "tables"
GROUND_TRUTH_CSV = "ground_truth.csv"

FIELDS = ["vendor", "invoice_number", "date", "amount", "currency", "gst_details"]
FUZZY_FIELDS = {"vendor": 0.80, "gst_details": 0.60}

# Published per-1M-token pricing (USD). Update these if provider pricing changes.
PRICING = {
    "gemini": {
        "input_per_1m": 0.50,
        "output_per_1m": 3.00,
    },
    "groq_qwen": {
        "input_per_1m": 0.60,
        "output_per_1m": 3.00,
    },
    "gpt": {
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
    },
    "claude": {
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
    },
}


def is_empty(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "n/a"):
        return True
    return False


def normalize_text(value) -> str:
    if is_empty(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalize_amount(value):
    if is_empty(value):
        return None
    text = str(value)
    text = re.sub(r"[^0-9.]", "", text)
    if text == "":
        return None
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def normalize_date(value):
    if is_empty(value):
        return None
    try:
        return dateparser.parse(str(value), dayfirst=True).date()
    except (ValueError, OverflowError):
        return None


def fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def score_field(field: str, predicted, truth) -> bool:
    """Returns True (correct) or False (incorrect) for one field."""

    if field in FUZZY_FIELDS:
        pred_norm = normalize_text(predicted)
        truth_norm = normalize_text(truth)
        if pred_norm == "" and truth_norm == "":
            return True
        if pred_norm == "" or truth_norm == "":
            return False
        # Containment check first: handles cases like gst_details where a
        # model correctly returns the GSTIN plus extra info (CGST/SGST/HSN)
        # that ground truth didn't capture - pure length-sensitive ratio
        # would unfairly penalize the extra (correct) detail. Guarded by a
        # minimum length so short strings don't trivially "contain" match.
        min_len_for_containment = 6
        if len(truth_norm) >= min_len_for_containment and truth_norm in pred_norm:
            return True
        if len(pred_norm) >= min_len_for_containment and pred_norm in truth_norm:
            return True
        return fuzzy_ratio(pred_norm, truth_norm) >= FUZZY_FIELDS[field]

    if field == "date":
        pred_date = normalize_date(predicted)
        truth_date = normalize_date(truth)
        if pred_date is None and truth_date is None:
            # Fall back to raw string comparison in case both are
            # unparseable but actually match (e.g. partial dates)
            return normalize_text(predicted) == normalize_text(truth)
        return pred_date == truth_date

    if field == "amount":
        pred_amt = normalize_amount(predicted)
        truth_amt = normalize_amount(truth)
        return pred_amt == truth_amt

    # invoice_number, currency: normalized exact match
    return normalize_text(predicted) == normalize_text(truth)


def load_ground_truth() -> dict:
    if not os.path.exists(GROUND_TRUTH_CSV):
        raise FileNotFoundError(f"{GROUND_TRUTH_CSV} not found - fill it in first")
    df = pd.read_csv(GROUND_TRUTH_CSV, dtype=str)
    return {row["bill_filename"]: row.to_dict() for _, row in df.iterrows()}


def load_extraction_results() -> list:
    results = []
    for path in glob.glob(os.path.join(OUTPUT_DIR, "*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def build_side_by_side_table(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide-format table for easy eyeballing: one row per (bill, field), with
    each model's prediction, whether it was correct, and the ground truth
    value all side by side. This is what you actually read through to spot-
    check specific disagreements, rather than the long detailed_scores.csv.
    """
    predicted_pivot = detailed_df.pivot_table(
        index=["bill_filename", "field"], columns="model", values="predicted", aggfunc="first"
    )
    correct_pivot = detailed_df.pivot_table(
        index=["bill_filename", "field"], columns="model", values="correct", aggfunc="first"
    )

    predicted_pivot.columns = [f"{col}_extracted" for col in predicted_pivot.columns]
    correct_pivot.columns = [f"{col}_correct" for col in correct_pivot.columns]

    ground_truth_series = detailed_df.groupby(["bill_filename", "field"])["ground_truth"].first()

    combined = pd.concat([ground_truth_series, predicted_pivot, correct_pivot], axis=1)
    combined = combined.reset_index()

    # Order fields consistently within each bill, and interleave
    # extracted/correct columns model by model for readability
    combined["field"] = pd.Categorical(combined["field"], categories=FIELDS, ordered=True)
    combined = combined.sort_values(["bill_filename", "field"])

    model_names = sorted(detailed_df["model"].unique())
    ordered_cols = ["bill_filename", "field", "ground_truth"]
    for model in model_names:
        ordered_cols += [f"{model}_extracted", f"{model}_correct"]
    ordered_cols = [c for c in ordered_cols if c in combined.columns]

    return combined[ordered_cols]


def build_detailed_scores(extraction_results: list, ground_truth: dict) -> pd.DataFrame:
    detailed_rows = []

    for record in extraction_results:
        bill_filename = record["bill_filename"]
        truth = ground_truth.get(bill_filename)
        if truth is None:
            print(f"  WARNING: no ground truth row for {bill_filename}, skipping")
            continue

        for model_name, model_result in record["results"].items():
            predicted_fields = model_result.get("fields", {})
            for field in FIELDS:
                correct = score_field(field, predicted_fields.get(field), truth.get(field))
                detailed_rows.append({
                    "bill_filename": bill_filename,
                    "model": model_name,
                    "field": field,
                    "predicted": predicted_fields.get(field),
                    "ground_truth": truth.get(field),
                    "correct": correct,
                })

    return pd.DataFrame(detailed_rows)


def compute_accuracy_table(detailed_df: pd.DataFrame) -> pd.DataFrame:
    accuracy_df = (
        detailed_df.groupby(["model", "field"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    accuracy_pivot = accuracy_df.pivot(index="field", columns="model", values="accuracy")
    accuracy_pivot = accuracy_pivot.reindex(FIELDS)

    return accuracy_pivot


def compute_cost_table(extraction_results: list) -> pd.DataFrame:
    rows = []
    for record in extraction_results:
        for model_name, model_result in record["results"].items():
            usage = model_result.get("usage", {})
            rows.append({
                "model": model_name,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("model").agg(
        bills_processed=("model", "count"),
        avg_input_tokens=("input_tokens", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
    ).reset_index()

    def cost_per_bill(row):
        pricing = PRICING.get(row["model"], {"input_per_1m": 0, "output_per_1m": 0})
        input_cost = (row["avg_input_tokens"] / 1_000_000) * pricing["input_per_1m"]
        output_cost = (row["avg_output_tokens"] / 1_000_000) * pricing["output_per_1m"]
        return input_cost + output_cost

    summary["cost_per_bill_usd"] = summary.apply(cost_per_bill, axis=1)
    summary["cost_per_100_bills_usd"] = summary["cost_per_bill_usd"] * 100

    return summary


def run():
    os.makedirs(TABLES_DIR, exist_ok=True)

    print("Loading ground truth...")
    ground_truth = load_ground_truth()

    print("Loading extraction results...")
    extraction_results = load_extraction_results()

    if not extraction_results:
        print(f"No results found in {OUTPUT_DIR}/. Run `python app.py` first.")
        return

    print("Scoring accuracy per model per field...")
    detailed_df = build_detailed_scores(extraction_results, ground_truth)
    detailed_df.to_csv(os.path.join(TABLES_DIR, "detailed_scores.csv"), index=False)

    accuracy_table = compute_accuracy_table(detailed_df)
    accuracy_table.to_csv(os.path.join(TABLES_DIR, "accuracy_by_model_field.csv"))
    print("\n=== Accuracy by field (proportion correct) ===")
    print(accuracy_table.round(2))

    print("\nBuilding side-by-side comparison table...")
    side_by_side = build_side_by_side_table(detailed_df)
    side_by_side.to_csv(os.path.join(TABLES_DIR, "side_by_side_comparison.csv"), index=False)
    print("\n=== Side-by-side (first bill only shown here, full table in tables/side_by_side_comparison.csv) ===")
    first_bill = side_by_side["bill_filename"].iloc[0]
    print(side_by_side[side_by_side["bill_filename"] == first_bill].to_string(index=False))

    print("\nComputing cost per model...")
    cost_table = compute_cost_table(extraction_results)
    cost_table.to_csv(os.path.join(TABLES_DIR, "cost_summary.csv"), index=False)
    print("\n=== Cost summary ===")
    print(cost_table.round(6))

    print(f"\nDone. Tables written to {TABLES_DIR}/")


if __name__ == "__main__":
    run()