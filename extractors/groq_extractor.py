"""
Groq vision extractor.

Uses Groq's free tier with the qwen/qwen3.6-27b multimodal model
(Groq's current vision-capable model as of mid-2026 — their earlier
Llama-vision-preview models were retired).
"""

import os
import time
import base64

from groq import Groq

from extractors.common import EXTRACTION_PROMPT, parse_model_json, build_result_record

MODEL_NAME = "qwen/qwen3.6-27b"


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment (.env)")
    return Groq(api_key=api_key)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract(image_path: str) -> dict:
    bill_filename = os.path.basename(image_path)
    client = _get_client()

    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
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
            max_completion_tokens=1536,
            # Qwen 3.6 defaults to "thinking mode", which was burning the
            # entire max_completion_tokens budget on <think> reasoning and
            # leaving no room for the actual JSON answer - every field came
            # back null even though the reasoning itself had the right
            # values. Disabling it here fixes that, and is also strictly
            # cheaper/faster since fewer output tokens get billed.
            reasoning_effort="none",
            reasoning_format="hidden",
        )
        latency = time.time() - start

        raw_text = completion.choices[0].message.content or ""
        parsed_fields = parse_model_json(raw_text)

        usage = completion.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return build_result_record(
            bill_filename=bill_filename,
            model_name="groq_qwen",
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
            model_name="groq_qwen",
            parsed_fields={field: None for field in
                            ["vendor", "invoice_number", "date", "amount", "currency", "gst_details"]},
            input_tokens=0,
            output_tokens=0,
            latency_seconds=latency,
            raw_response="",
            error=str(e),
        )