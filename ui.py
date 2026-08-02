"""
Minimal Streamlit UI for the bill extraction project.

Tabs:
  1. Run Extraction — upload a bill image, pick which models to run, see
     each model's extracted fields side by side.
  2. Push to Zoho    — pick an already-extracted bill + model, review the
     expense payload, and create it in Zoho Books.
  3. Accuracy        — per-field accuracy by model, from compare.py.
  4. Cost            — per-model cost per bill / per 100 bills, from compare.py.
  5. Compare         — full ground-truth vs. prediction table, with the
     bill image shown alongside so a judge can eyeball disagreements.

This is a thin viewer/runner on top of the existing pipeline — it reuses
the same extractor functions app.py calls, the same tables compare.py
writes, and the same Zoho functions zoho_upload.py uses. No logic is
duplicated.

Run with:
    streamlit run ui.py
"""

import os
import json
import glob
import tempfile
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from dateutil import parser as dateparser

load_dotenv()

from extractors import gemini_extractor, groq_extractor, gpt_extractor,claude_extractor

from zoho_upload import (
    get_access_token,
    list_chart_of_accounts,
    build_expense_payload,
    create_expense,
)

EXTRACTORS = {
    "gemini": gemini_extractor,
    "groq_qwen": groq_extractor,
    "gpt": gpt_extractor,
    "claude": claude_extractor,
}

BILLS_DIR = "Bills"
OUTPUT_DIR = "output"
TABLES_DIR = "tables"
FIELDS = ["vendor", "invoice_number", "date", "amount", "currency", "gst_details"]

st.set_page_config(page_title="Bill Extraction Comparison", layout="wide")

st.markdown(
    """
    <style>
    @keyframes scroll-hint-pulse {
        0%   { transform: translateY(0px);   opacity: 1;   }
        50%  { transform: translateY(8px);   opacity: 0.35; }
        100% { transform: translateY(0px);   opacity: 1;   }
    }
    .scroll-hint-arrow {
        position: fixed;
        top: 60px;
        right: 28px;
        z-index: 9999;
        font-size: 26px;
        line-height: 1;
        color: #ff4b4b;
        animation: scroll-hint-pulse 1.5s ease-in-out infinite;
        pointer-events: none;
        user-select: none;
    }
    </style>
    <div class="scroll-hint-arrow">&#8595;</div>
    """,
    unsafe_allow_html=True,
)

st.title("Handwritten Bill Extraction — Model Comparison")

tab_run, tab_zoho, tab_accuracy, tab_cost, tab_compare = st.tabs(
    ["Run Extraction", "Push to Zoho", "Accuracy", "Cost", "Compare"]
)


def validate_bill_fields(fields: dict):
    """
    Pre-flight checks run in the UI before anything is sent to Zoho.

    Returns (errors, warnings):
      - errors   block the "Create expense" button (Zoho would reject the
                 request anyway - better to catch it here with a clear
                 message than after a failed API call).
      - warnings are shown but don't block submission (e.g. missing GST,
        which is legitimate for plenty of small-shop bills).
    """
    errors = []
    warnings = []

    # --- date ---
    raw_date = fields.get("date")
    if not raw_date:
        errors.append("No date extracted — Zoho requires a date for every expense.")
    else:
        try:
            parsed_date = dateparser.parse(str(raw_date), dayfirst=True).date()
            if parsed_date > date.today():
                warnings.append(f"Extracted date ({parsed_date}) is in the future — double-check the bill.")
            if parsed_date < date.today() - timedelta(days=365 * 15):
                warnings.append(f"Extracted date ({parsed_date}) is over 15 years old — double-check the bill.")
        except (ValueError, OverflowError):
            errors.append(f"Couldn't parse '{raw_date}' as a date — Zoho would reject this.")

    # --- amount ---
    raw_amount = fields.get("amount")
    if raw_amount in (None, "", "null"):
        errors.append("No amount extracted — Zoho requires an amount for every expense.")
    else:
        try:
            amount_value = float(raw_amount)
            if amount_value <= 0:
                errors.append(f"Amount ({amount_value}) must be greater than zero.")
            elif amount_value > 500_000:
                warnings.append(
                    f"Amount (₹{amount_value:,.2f}) looks unusually high for a single bill — double-check the extraction."
                )
        except (TypeError, ValueError):
            errors.append(f"Amount '{raw_amount}' isn't a valid number — Zoho would reject this.")

    # --- soft checks, non-blocking ---
    if not fields.get("vendor"):
        warnings.append("No vendor name extracted — the expense description will say 'Unknown vendor'.")
    if not fields.get("invoice_number"):
        warnings.append("No invoice/bill number extracted — reference_number will be left blank.")
    if not fields.get("gst_details"):
        warnings.append("No GST details extracted — fine if the bill genuinely has none.")

    return errors, warnings


def find_bill_image(bill_filename: str):
    """bill_filename is a base name like 'Bill_1' or 'Bill_1.jpg' - find the
    actual image on disk regardless of extension."""
    stem = os.path.splitext(bill_filename)[0]
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG"):
        candidate = os.path.join(BILLS_DIR, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


# ----------------------------------------------------------------------
# Tab 1: upload a bill, run selected extractors, show fields side by side
# ----------------------------------------------------------------------
with tab_run:
    st.subheader("Upload a bill and compare model extractions")

    uploaded_file = st.file_uploader(
        "Upload a bill/receipt image", type=["png", "jpg", "jpeg", "webp"]
    )

    selected_models = st.multiselect(
        "Models to run",
        options=list(EXTRACTORS.keys()),
        default=list(EXTRACTORS.keys()),
    )

    run_clicked = st.button("Run extraction", type="primary", disabled=uploaded_file is None)

    if uploaded_file is not None:
        st.image(uploaded_file, caption=uploaded_file.name, width=350)

    if run_clicked and uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        results = {}
        cols = st.columns(len(selected_models)) if selected_models else []

        for col, model_name in zip(cols, selected_models):
            with col:
                st.markdown(f"**{model_name}**")
                with st.spinner(f"Calling {model_name}..."):
                    result = EXTRACTORS[model_name].extract(tmp_path)
                results[model_name] = result

                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.json(result["fields"])
                    st.caption(
                        f"{result['latency_seconds']:.1f}s · "
                        f"{result['usage']['input_tokens']} in / "
                        f"{result['usage']['output_tokens']} out tokens"
                    )

        os.unlink(tmp_path)

        if results:
            st.markdown("---")
            st.markdown("**Side-by-side fields**")
            table_rows = {
                model_name: {f: res.get("fields", {}).get(f) for f in FIELDS}
                for model_name, res in results.items()
            }
            st.dataframe(pd.DataFrame(table_rows).T, use_container_width=True)

    st.caption(
        "Note: this tab has no ground truth for a freshly uploaded image, "
        "so it only shows raw predictions, not correctness. Scored results "
        "are in the Accuracy / Compare tabs."
    )


# ----------------------------------------------------------------------
# Tab 2: push an already-extracted bill into Zoho Books as an expense
# ----------------------------------------------------------------------
with tab_zoho:
    st.subheader("Create a Zoho Books expense from an extracted bill")

    missing_env = [
        var for var in
        ["ZOHO_ORGANIZATION_ID", "ZOHO_REFRESH_TOKEN", "ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET"]
        if not os.environ.get(var)
    ]
    if missing_env:
        st.warning(
            f"Missing from .env: {', '.join(missing_env)}. "
            "Fill these in before pushing to Zoho."
        )
    else:
        bill_json_paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
        if not bill_json_paths:
            st.info("No extraction results yet. Run `python app.py` first.")
        else:
            bill_choice = st.selectbox(
                "Bill", [os.path.basename(p) for p in bill_json_paths]
            )
            with open(os.path.join(OUTPUT_DIR, bill_choice)) as f:
                combined = json.load(f)
            available_models = [
                m for m, r in combined["results"].items() if not r.get("error")
            ]
            model_choice = st.selectbox("Model's extraction to use", available_models)

            fields = combined["results"][model_choice]["fields"]
            st.markdown("**Extracted fields**")
            st.json(fields)

            bill_image = find_bill_image(combined["bill_filename"])
            if bill_image:
                st.image(bill_image, width=300)

            # --- Pre-flight safety checks, shown before any API call ---
            st.markdown("**Pre-flight checks**")
            errors, warnings = validate_bill_fields(fields)

            if not errors and not warnings:
                st.success("Looks good — no issues found.")
            for err in errors:
                st.error(err)
            for warn in warnings:
                st.warning(warn)

            if errors:
                st.info("Fix the issue above (e.g. by re-running extraction, or trying a different model's result for this bill) before pushing to Zoho.")

            # Fetch chart of accounts once per session so re-runs don't
            # keep hitting the Zoho API
            if "zoho_accounts" not in st.session_state:
                if st.button("Load Zoho accounts"):
                    with st.spinner("Fetching chart of accounts from Zoho..."):
                        try:
                            token = get_access_token()
                            accounts = list_chart_of_accounts(token)
                            st.session_state["zoho_token"] = token
                            st.session_state["zoho_accounts"] = accounts
                        except Exception as e:
                            st.error(f"Couldn't fetch accounts: {e}")

            if "zoho_accounts" in st.session_state:
                accounts = st.session_state["zoho_accounts"]
                expense_accounts = {
                    a["account_name"]: a["account_id"]
                    for a in accounts if a.get("account_type") == "expense"
                }
                cash_accounts = {
                    a["account_name"]: a["account_id"]
                    for a in accounts if a.get("account_type") in ("cash", "bank")
                }

                expense_name = st.selectbox("Expense account", list(expense_accounts.keys()))
                paid_through_name = st.selectbox("Paid-through account", list(cash_accounts.keys()))

                if st.button("Create expense in Zoho Books", type="primary", disabled=bool(errors)):
                    try:
                        payload = build_expense_payload(
                            fields,
                            account_id=expense_accounts[expense_name],
                            paid_through_account_id=cash_accounts[paid_through_name],
                        )

                        # Safety net: the Zoho Expenses endpoint has no
                        # `gst_no` field (that field only exists on
                        # Contacts) - sending it is a guaranteed 400. This
                        # strips it here regardless of what
                        # build_expense_payload() returns, so a stale/
                        # unpatched zoho_upload.py can't silently break
                        # the request. The GSTIN is preserved in the
                        # description instead of being dropped.
                        INVALID_EXPENSE_FIELDS = ["gst_no"]
                        removed_fields = {}
                        for field_name in INVALID_EXPENSE_FIELDS:
                            if field_name in payload:
                                removed_fields[field_name] = payload.pop(field_name)

                        if "gst_no" in removed_fields:
                            gstin_value = removed_fields["gst_no"]
                            if gstin_value and gstin_value not in payload.get("description", ""):
                                payload["description"] = (
                                    f"{payload.get('description', '')} (GSTIN: {gstin_value})".strip()
                                )
                            st.warning(
                                "Removed 'gst_no' before sending — Zoho's Expenses endpoint "
                                "doesn't accept it (that field only exists on Contacts)."
                            )

                        st.markdown("**Payload sent to Zoho**")
                        st.json(payload)
                        with st.spinner("Creating expense..."):
                            result = create_expense(st.session_state["zoho_token"], payload)
                        expense_id = result.get("expense", {}).get("expense_id")
                        st.success(f"Expense created — expense_id: {expense_id}")
                    except Exception as e:
                        st.error(f"Failed to create expense: {e}")


# ----------------------------------------------------------------------
# Tab 3: accuracy by model / field
# ----------------------------------------------------------------------
with tab_accuracy:
    st.subheader("Accuracy by field")

    accuracy_path = os.path.join(TABLES_DIR, "accuracy_by_model_field.csv")
    if not os.path.exists(accuracy_path):
        st.info(
            "No results yet. Run `python app.py` then `python compare.py` "
            "first, then reload this page."
        )
    else:
        accuracy_df = pd.read_csv(accuracy_path, index_col=0)

        st.dataframe(accuracy_df.style.format("{:.0%}", na_rep="—"), use_container_width=True)
        st.bar_chart(accuracy_df)

        st.markdown("**Overall accuracy per model** (averaged across all fields)")
        overall = accuracy_df.mean().sort_values(ascending=False)
        st.bar_chart(overall)


# ----------------------------------------------------------------------
# Tab 4: cost per model
# ----------------------------------------------------------------------
with tab_cost:
    st.subheader("Cost per model")

    cost_path = os.path.join(TABLES_DIR, "cost_summary.csv")
    if not os.path.exists(cost_path):
        st.info("No results yet. Run `python app.py` then `python compare.py` first.")
    else:
        cost_df = pd.read_csv(cost_path)
        st.dataframe(cost_df, use_container_width=True)

        chart_df = cost_df.set_index("model")[["cost_per_100_bills_usd"]]
        st.markdown("**Cost per 100 bills (USD)**")
        st.bar_chart(chart_df)

        chart_df2 = cost_df.set_index("model")[["avg_input_tokens", "avg_output_tokens"]]
        st.markdown("**Average tokens per bill**")
        st.bar_chart(chart_df2)

        st.caption(
            "Pricing is hardcoded in compare.py's PRICING dict — double-check "
            "against current provider pricing pages before treating these as final."
        )


# ----------------------------------------------------------------------
# Tab 5: full compare table, with the bill image alongside it
# ----------------------------------------------------------------------
with tab_compare:
    st.subheader("Predictions vs. ground truth")

    side_by_side_path = os.path.join(TABLES_DIR, "side_by_side_comparison.csv")
    if not os.path.exists(side_by_side_path):
        st.info("No results yet. Run `python app.py` then `python compare.py` first.")
    else:
        side_by_side_df = pd.read_csv(side_by_side_path)
        bill_options = sorted(side_by_side_df["bill_filename"].unique())

        selected_bill = st.selectbox("Bill", bill_options)

        image_col, table_col = st.columns([1, 2])
        with image_col:
            bill_image = find_bill_image(selected_bill)
            if bill_image:
                st.image(bill_image, use_container_width=True)
            else:
                st.info(f"Image not found in {BILLS_DIR}/ for {selected_bill}")

        with table_col:
            bill_df = side_by_side_df[side_by_side_df["bill_filename"] == selected_bill]
            bill_df = bill_df.drop(columns=["bill_filename"])

            # Highlight incorrect cells so mismatches jump out visually
            correct_cols = [c for c in bill_df.columns if c.endswith("_correct")]

            def highlight_incorrect(row):
                styles = [""] * len(row)
                for i, col in enumerate(row.index):
                    if col in correct_cols and row[col] is False:
                        styles[i] = "background-color: #ffe0e0"
                return styles

            st.dataframe(
                bill_df.style.apply(highlight_incorrect, axis=1),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.markdown("**All bills**")
        st.dataframe(side_by_side_df, use_container_width=True)