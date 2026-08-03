"""
GPT vision extractor.

Uses OpenRouter (https://openrouter.ai) as the gateway to reach OpenAI's
GPT-4o-mini - one API key, OpenAI-compatible endpoint, no separate OpenAI
account/billing needed. Check https://openrouter.ai/models for current
pricing on this model before trusting compare.py's cost numbers - it
changes over time.
"""

import os
import time
import base64

from openai import OpenAI

from extractors.common import EXTRACTION_PROMPT, parse_model_json, build_result_record,get_image_mime_type

MODEL_NAME = "openai/gpt-4o-mini"


def _get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment (.env)")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract(image_path: str) -> dict:
    bill_filename = os.path.basename(image_path)
    client = _get_client()

    mime_type = get_image_mime_type(image_path)
    base64_image = _encode_image(image_path)

    start = time.time()
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=1024,
            extra_headers={
                "HTTP-Referer": "https://github.com/",
                "X-Title": "Handwritten Bill Extraction Eval",
            },
        )
        latency = time.time() - start

        raw_text = completion.choices[0].message.content or ""
        parsed_fields = parse_model_json(raw_text)

        usage = completion.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return build_result_record(
            bill_filename=bill_filename,
            model_name="gpt",
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
            model_name="gpt",
            parsed_fields={field: None for field in
                            ["vendor", "invoice_number", "date", "amount", "currency", "gst_details"]},
            input_tokens=0,
            output_tokens=0,
            latency_seconds=latency,
            raw_response="",
            error=str(e),
        )