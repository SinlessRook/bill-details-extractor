"""
Pushes extracted bill data into Zoho Books as expense entries.

SETUP (one-time):
1. Run:
       python zoho_upload.py --list-accounts
   This prints your Chart of Accounts. You need two account_ids:
     - an EXPENSE account (e.g. "Office Supplies", "Travel", "Miscellaneous")
     - a PAID-THROUGH account, i.e. where the money came from (e.g. "Petty
       Cash", "Cash", or your bank account) - Zoho calls this
       paid_through_account_id and requires it on every expense.
   Both are org-specific numeric IDs, not guessable, hence this step.

USAGE:
    python zoho_upload.py --bill Bill_1.json --model gemini \
        --account-id 1234000000012345 --paid-through-id 1234000000012399

    --bill is the filename inside output/ (as produced by app.py)
    --model is which extractor's values to trust: "gemini" or "groq_qwen"

AUTH: uses ZOHO_REFRESH_TOKEN from .env to mint a fresh access token on
every run (access tokens expire in ~1 hour; the refresh token itself
doesn't expire unless you revoke it in Zoho).

NOTE ON REGION: ZOHO_ACCOUNTS_BASE and ZOHO_API_BASE in .env must match
the data center your Zoho Books org lives in - .in for India accounts
(most likely, given GSTIN bills), .com for US, etc. Wrong region is a
common source of "invalid client" / "invalid organization" errors.
"""

import os
import re
import json
import argparse

import requests
from dateutil import parser as dateparser
from dotenv import load_dotenv

load_dotenv()

ACCOUNTS_BASE = os.environ.get("ZOHO_ACCOUNTS_BASE", "https://accounts.zoho.in")
API_BASE = os.environ.get("ZOHO_API_BASE", "https://www.zohoapis.in/books/v3")
ORG_ID = os.environ.get("ZOHO_ORGANIZATION_ID")

OUTPUT_DIR = "output"

GSTIN_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b")


def get_access_token() -> str:
    """Exchange the long-lived refresh token for a short-lived access token."""
    resp = requests.post(
        f"{ACCOUNTS_BASE}/oauth/v2/token",
        data={
            "refresh_token": os.environ["ZOHO_REFRESH_TOKEN"],
            "client_id": os.environ["ZOHO_CLIENT_ID"],
            "client_secret": os.environ["ZOHO_CLIENT_SECRET"],
            "grant_type": "refresh_token",
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if "access_token" not in data:
        raise RuntimeError(f"Failed to get Zoho access token: {data}")
    return data["access_token"]


def auth_headers(access_token: str) -> dict:
    return {
        "Authorization": f"Zoho-oauthtoken {access_token}",
        "content-type": "application/json",
    }


def list_chart_of_accounts(access_token: str, filter_by: str = None):
    params = {"organization_id": ORG_ID}
    if filter_by:
        params["filter_by"] = filter_by

    resp = requests.get(
        f"{API_BASE}/chartofaccounts", headers=auth_headers(access_token), params=params
    )
    resp.raise_for_status()
    return resp.json().get("chartofaccounts", [])


def print_accounts_for_selection(access_token: str):
    print("=== EXPENSE accounts (pick one for --account-id) ===")
    for acc in list_chart_of_accounts(access_token, filter_by="AccountType.Expense"):
        print(f"  {acc['account_id']:<20} {acc['account_name']}")

    print("\n=== CASH / BANK accounts (pick one for --paid-through-id) ===")
    all_accounts = list_chart_of_accounts(access_token)
    for acc in all_accounts:
        if acc.get("account_type") in ("cash", "bank"):
            print(f"  {acc['account_id']:<20} {acc['account_name']}  ({acc['account_type']})")


def extract_gstin(gst_details) -> str:
    """Pull a 15-char GSTIN out of a free-text gst_details string, if present."""
    if not gst_details:
        return None
    match = GSTIN_PATTERN.search(str(gst_details).upper())
    return match.group(0) if match else None


def normalize_date_for_zoho(date_str) -> str:
    """Zoho expects YYYY-MM-DD."""
    parsed = dateparser.parse(str(date_str), dayfirst=True)
    return parsed.strftime("%Y-%m-%d")


def load_bill_result(bill_json_filename: str, model_name: str) -> dict:
    path = os.path.join(OUTPUT_DIR, bill_json_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - check the filename matches output/*.json")

    with open(path) as f:
        combined = json.load(f)

    result = combined["results"].get(model_name)
    if result is None:
        raise ValueError(f"No '{model_name}' result in {path}. Available: {list(combined['results'].keys())}")
    if result.get("error"):
        raise ValueError(f"{model_name} extraction for this bill errored: {result['error']}")

    return result["fields"]


def build_expense_payload(fields: dict, account_id: str, paid_through_account_id: str) -> dict:
    if not fields.get("date"):
        raise ValueError("No date extracted for this bill - can't create expense without one")
    if not fields.get("amount"):
        raise ValueError("No amount extracted for this bill - can't create expense without one")

    payload = {
        "account_id": account_id,
        "paid_through_account_id": paid_through_account_id,
        "date": normalize_date_for_zoho(fields["date"]),
        "amount": float(fields["amount"]),
        "reference_number": fields.get("invoice_number") or "",
        "description": f"{fields.get('vendor') or 'Unknown vendor'} - auto-extracted from handwritten bill",
    }

    gstin = extract_gstin(fields.get("gst_details"))
    if gstin:
        payload["gst_no"] = gstin

    return payload


def create_expense(access_token: str, payload: dict) -> dict:
    resp = requests.post(
        f"{API_BASE}/expenses",
        headers=auth_headers(access_token),
        params={"organization_id": ORG_ID},
        json=payload,
    )
    data = resp.json()
    if resp.status_code >= 400 or data.get("code", 0) != 0:
        raise RuntimeError(f"Zoho API error ({resp.status_code}): {data}")
    return data


def run():
    parser = argparse.ArgumentParser(description="Push an extracted bill into Zoho Books as an expense.")
    parser.add_argument("--list-accounts", action="store_true",
                         help="List Chart of Accounts to find account_id/paid_through_account_id, then exit.")
    parser.add_argument("--bill", help="Filename inside output/, e.g. Bill_1.json")
    parser.add_argument("--model", choices=["gemini", "groq_qwen","gpt", "claude"], help="Which model's extraction to use")
    parser.add_argument("--account-id", help="Expense account_id from --list-accounts")
    parser.add_argument("--paid-through-id", help="Paid-through account_id from --list-accounts")
    args = parser.parse_args()

    if not ORG_ID:
        raise RuntimeError("ZOHO_ORGANIZATION_ID not set in .env")

    print("Getting Zoho access token...")
    access_token = get_access_token()

    if args.list_accounts:
        print_accounts_for_selection(access_token)
        return

    if not (args.bill and args.model and args.account_id and args.paid_through_id):
        parser.error("--bill, --model, --account-id, and --paid-through-id are all required "
                     "(unless using --list-accounts)")

    print(f"Loading {args.model} extraction for {args.bill}...")
    fields = load_bill_result(args.bill, args.model)
    print(f"  Extracted fields: {fields}")

    payload = build_expense_payload(fields, args.account_id, args.paid_through_id)
    print(f"  Expense payload: {payload}")

    print("Creating expense in Zoho Books...")
    result = create_expense(access_token, payload)
    expense_id = result.get("expense", {}).get("expense_id")
    print(f"Done. Expense created (expense_id: {expense_id})")


if __name__ == "__main__":
    run()