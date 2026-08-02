"""
Main entry point.

Loops over every image in Bills/, runs it through both extractors
(Gemini + Groq), and writes one JSON file per bill into output/
containing both models' results side by side.

By default this is CACHED: if output/<bill>.json already exists and a
given model's result in it succeeded (no error), that model is skipped
for that bill - only missing/errored results get (re-)called. This is
what you want when you're rate-limited and running app.py multiple
times as you add bills or retry failures.

Usage:
    python app.py              # cached run - skip already-successful results
    python app.py --no-cache   # force re-run every bill on every model
"""

import os
import json
import glob
import argparse

from dotenv import load_dotenv

load_dotenv()

from extractors import gemini_extractor, groq_extractor, gpt_extractor,claude_extractor

BILLS_DIR = "Bills"
OUTPUT_DIR = "output"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

EXTRACTORS = {
    "gemini": gemini_extractor,
    "groq_qwen": groq_extractor,
    "gpt": gpt_extractor,
    "claude": claude_extractor,  # add new extractors here as you build them
}


def get_bill_paths():
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(BILLS_DIR, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(BILLS_DIR, f"*{ext.upper()}")))
    return sorted(set(paths))


def load_existing_result(out_path):
    """Returns the existing combined dict for this bill, or None if no
    cached file exists / it can't be read."""
    if not os.path.exists(out_path):
        return None
    try:
        with open(out_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def run(use_cache: bool):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bill_paths = get_bill_paths()

    if not bill_paths:
        print(f"No images found in {BILLS_DIR}/. Add your bill images there first.")
        return

    mode = "cached (skipping already-successful results)" if use_cache else "no-cache (re-running everything)"
    print(f"Found {len(bill_paths)} bills. Mode: {mode}\n")

    for path in bill_paths:
        bill_filename = os.path.basename(path)
        out_name = os.path.splitext(bill_filename)[0] + ".json"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        existing = load_existing_result(out_path) if use_cache else None
        existing_results = existing["results"] if existing else {}

        print(f"Processing {bill_filename}...")
        combined_results = {}

        for model_name, extractor in EXTRACTORS.items():
            cached = existing_results.get(model_name)
            if cached is not None and not cached.get("error"):
                print(f"  {model_name}: SKIPPED (cached, already succeeded)")
                combined_results[model_name] = cached
                continue

            result = extractor.extract(path)
            if result["error"]:
                print(f"  {model_name}: ERROR - {result['error']}")
            else:
                print(f"  {model_name}: OK ({result['latency_seconds']:.1f}s)")
            combined_results[model_name] = result

        combined = {
            "bill_filename": bill_filename,
            "results": combined_results,
        }

        with open(out_path, "w") as f:
            json.dump(combined, f, indent=2)

    print(f"\nDone. Results written to {OUTPUT_DIR}/")
    print("Next: fill in ground_truth.csv, then run `python compare.py`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bill extraction across Gemini and Groq.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-extraction of every bill/model, ignoring any existing output/*.json files.",
    )
    args = parser.parse_args()
    run(use_cache=not args.no_cache)