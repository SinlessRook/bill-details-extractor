"""
Shared logic used by every model-specific extractor.

Keeping the prompt and schema in ONE place is deliberate: it's what makes
this a fair comparison. The only thing that changes between extractors is
which API is being called — the instructions given to the model are
identical.
"""

import json
import re
from PIL import Image

# The exact fields every extractor must return.
SCHEMA_FIELDS = [
    "vendor",
    "invoice_number",
    "date",
    "amount",
    "currency",
    "gst_details",
]

EXTRACTION_PROMPT = """You are looking at a photo of a handwritten or printed bill/receipt from India.

Extract the following fields and return ONLY a JSON object with exactly these keys:
- vendor: the shop/business name as written on the bill
- invoice_number: the bill/invoice number if present, otherwise null
- date: the date on the bill, in DD-MM-YYYY format if you can determine it, otherwise the raw text you see
- amount: the total amount as a plain number (no currency symbol, no commas), otherwise null
- currency: the currency code, e.g. "INR", if not specified assume "INR"
- gst_details: any GST number, tax rate, or tax amount visible on the bill, as free text. If nothing tax-related is visible, use null.

Rules:
- Return ONLY the JSON object, no markdown code fences, no explanation, no extra text.
- If a field is illegible or not present, use null for that field — do not guess or invent values.
- Do not include any commentary before or after the JSON.
"""



def get_image_mime_type(image_path: str) -> str:
    """
    Detect the actual image MIME type from the file contents.

    Supports JPEG, PNG, WEBP, GIF, BMP and TIFF.
    Falls back to image/jpeg if the format is unknown.
    """
    with Image.open(image_path) as img:
        fmt = img.format.upper()

    mapping = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
    }

    return mapping.get(fmt, "image/jpeg")

def parse_model_json(raw_text: str) -> dict:
    """
    Models sometimes wrap JSON in ```json fences or add stray text.
    This strips that and returns a dict with all SCHEMA_FIELDS present
    (missing ones filled with None) so downstream code never has to
    special-case a missing key.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to find the first {...} block in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    # Guarantee every expected field exists, even if the model omitted it
    result = {field: parsed.get(field) for field in SCHEMA_FIELDS}
    return result


def build_result_record(bill_filename: str, model_name: str, parsed_fields: dict,
                         input_tokens: int, output_tokens: int, latency_seconds: float,
                         raw_response: str, error: str = None) -> dict:
    """
    Standard wrapper around every extraction result so app.py and compare.py
    don't need to know anything model-specific.
    """
    return {
        "bill_filename": bill_filename,
        "model": model_name,
        "fields": parsed_fields,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "latency_seconds": latency_seconds,
        "raw_response": raw_response,
        "error": error,
    }
