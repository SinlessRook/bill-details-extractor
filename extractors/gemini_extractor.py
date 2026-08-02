"""
Gemini vision extractor.

Uses Google AI Studio's free-tier Gemini API to extract bill fields from
an image.
"""

import os
import time

from google import genai
from google.genai import types

from extractors.common import EXTRACTION_PROMPT, parse_model_json, build_result_record

MODEL_NAME = "gemini-3-flash-preview"

def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment (.env)")
    return genai.Client(api_key=api_key)


def extract(image_path: str) -> dict:
    """
    Run extraction on a single image. Returns a result dict built by
    build_result_record — never raises, errors are captured in the record
    so a single bad bill/API call doesn't crash the whole batch run.
    """
    bill_filename = os.path.basename(image_path)
    client = _get_client()

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    start = time.time()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                EXTRACTION_PROMPT,
            ],
        )
        latency = time.time() - start

        raw_text = response.text or ""
        parsed_fields = parse_model_json(raw_text)

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        return build_result_record(
            bill_filename=bill_filename,
            model_name="gemini",
            parsed_fields=parsed_fields,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            raw_response=raw_text,
        )

    except Exception as e:
        latency = time.time() - start
        return build_result_record(
            bill_filename=bill_filename,
            model_name="gemini",
            parsed_fields={field: None for field in
                            ["vendor", "invoice_number", "date", "amount", "currency", "gst_details"]},
            input_tokens=0,
            output_tokens=0,
            latency_seconds=latency,
            raw_response="",
            error=str(e),
        )