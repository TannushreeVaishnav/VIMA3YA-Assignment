# audit_and_clean.py
# Data cleaning & audit script for NimbusPay ticket triage
# Written for VIMA3YA ML Intern assignment

import json
import csv
import re
from collections import Counter

# Ground truth schema constants from SCHEMA.md
KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
CATEGORIES = {"payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other"}
PRIORITIES = {"P1", "P2", "P3"}
CHANNELS = {"upi", "card", "netbanking", "wallet"}
LANGUAGES = {"en", "hi-en"}

# Regex helpers for rule checks
LEGAL_RE = re.compile(r"\b(rbi|ombudsman|consumer\s+court|lawyer|legal\s+notice|legal\s+action)\b", re.I)
REPEAT_RE = re.compile(r"\b(third\s+time|again\s+and\s+again|baar\s+baar|already\s+contacted|multiple\s+times|several\s+times|contacted\s+before|second\s+time|4th\s+time|fourth\s+time|pehle\s+bhi|already\s+written)\b", re.I)
TXN_ID_RE = re.compile(r"^NP\d{10}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DMY_DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")

def audit_and_clean(train_path="candidate_pack/data/train.jsonl",
                    cleaned_path="candidate_pack/data/train_cleaned.jsonl",
                    log_path="cleaning_log.csv"):
    
    with open(train_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    log_entries = []  # tuples of (id, action, reason)
    cleaned_rows = []
    seen_tickets = {}

    print(f"Loaded {len(rows)} training examples. Starting audit...")

    for r in rows:
        row_id = r["id"]
        messages = r.get("messages", [])
        
        # 1. Structure check: system, user, assistant
        if len(messages) != 3 or [m["role"] for m in messages] != ["system", "user", "assistant"]:
            log_entries.append((row_id, "drop", "Malformed message roles or count"))
            continue

        user_text = messages[1]["content"]
        # extract body without header lines
        body_lines = [l for l in user_text.split("\n") 
                      if not l.startswith("Received:") and not l.startswith("Source:")]
        user_body = "\n".join(body_lines).strip()

        # 2. Check for synthetic dummy/spam tickets
        if user_body.lower() in {"test", "test test 123", "asdf asdf", ""}:
            log_entries.append((row_id, "drop", f"Synthetic/spam ticket with fake labels: '{user_body}'"))
            continue

        # 3. Drop exact duplicates to avoid overweighting
        user_key = user_text.strip()
        if user_key in seen_tickets:
            first_id = seen_tickets[user_key]
            log_entries.append((row_id, "drop", f"Duplicate customer message of {first_id}"))
            continue
        seen_tickets[user_key] = row_id

        # 4. Parse assistant JSON output
        raw_target = messages[2]["content"]
        try:
            label = json.loads(raw_target)
        except Exception as err:
            # truncated strings cut off mid-sentence
            log_entries.append((row_id, "drop", f"Truncated unparseable JSON: {str(err)}"))
            continue

        if not isinstance(label, dict):
            log_entries.append((row_id, "drop", "Assistant message is not a JSON object"))
            continue

        fixes = []

        # Fix legacy 'prio' key from v1 schema
        if "prio" in label and "priority" not in label:
            old_prio = label.pop("prio")
            prio_map = {"low": "P3", "medium": "P2", "high": "P1", "P1": "P1", "P2": "P2", "P3": "P3"}
            label["priority"] = prio_map.get(old_prio, "P3")
            fixes.append(f"Mapped legacy key 'prio': '{old_prio}' to 'priority': '{label['priority']}'")

        # Ensure all 8 required keys exist
        for k in KEYS:
            if k not in label:
                label[k] = None
                fixes.append(f"Filled missing key '{k}' with null")

        # Strip unauthorized extra keys
        extra = [k for k in label if k not in KEYS]
        for k in extra:
            del label[k]
            fixes.append(f"Removed extra key '{k}'")

        # Normalize txn_id (uppercase NP + 10 digits)
        tid = label.get("txn_id")
        if tid is not None:
            if not TXN_ID_RE.match(tid):
                clean_tid = re.sub(r"[^a-zA-Z0-9]", "", tid).upper()
                if TXN_ID_RE.match(clean_tid):
                    fixes.append(f"Normalized txn_id '{tid}' -> '{clean_tid}'")
                    label["txn_id"] = clean_tid
                else:
                    fixes.append(f"Invalid txn_id '{tid}' set to null")
                    label["txn_id"] = None

        # Convert DD/MM/YYYY dates to ISO YYYY-MM-DD
        tdate = label.get("txn_date")
        if tdate is not None:
            dmy = DMY_DATE_RE.match(tdate)
            if dmy:
                d, m, y = dmy.groups()
                iso_d = f"{y}-{m}-{d}"
                fixes.append(f"Converted txn_date '{tdate}' -> '{iso_d}'")
                label["txn_date"] = iso_d
            elif not DATE_RE.match(tdate):
                fixes.append(f"Malformed date '{tdate}' set to null")
                label["txn_date"] = None

        # Check amount: raw rupees mistakenly stored instead of paise
        amt = label.get("amount_paise")
        if amt is not None:
            rupee_matches = re.findall(r"(?:Rs\.?|rupees|INR)\s*([\d,]+(?:\.\d+)?)", user_text, re.I)
            rupee_matches += re.findall(r"([\d,]+(?:\.\d+)?)\s*(?:rupees|INR)", user_text, re.I)
            for m in rupee_matches:
                try:
                    val = float(m.replace(",", ""))
                    if val > 0 and amt == int(val):
                        paise_val = int(round(val * 100))
                        fixes.append(f"Converted rupees ({amt}) to paise ({paise_val})")
                        label["amount_paise"] = paise_val
                        amt = paise_val
                        break
                except:
                    pass

        # Transaction fields must be null for non-txn categories
        cat = label.get("category")
        if cat in {"account_access", "kyc", "other"}:
            for field in ["amount_paise", "txn_date", "txn_id", "channel"]:
                if label.get(field) is not None:
                    fixes.append(f"Reset '{field}' to null for non-transaction category '{cat}'")
                    label[field] = None
            amt = None

        # Fix priority rule hierarchy per SCHEMA.md
        is_legal = bool(LEGAL_RE.search(user_text))
        is_fraud = (cat == "fraud")
        is_large = (amt is not None and amt >= 5000000)
        
        if is_fraud or is_large or is_legal:
            target_prio = "P1"
        else:
            is_p2 = ((cat in {"payment_failed", "refund"} and amt is not None and amt >= 500000) or (cat == "account_access"))
            target_prio = "P2" if is_p2 else "P3"

        if label.get("priority") != target_prio:
            fixes.append(f"Corrected priority '{label.get('priority')}' -> '{target_prio}'")
            label["priority"] = target_prio

        # Fix needs_human rule (true if P1 or repeated contact)
        is_repeat = bool(REPEAT_RE.search(user_text))
        target_nh = True if (label["priority"] == "P1" or is_repeat) else False
        if label.get("needs_human") != target_nh:
            fixes.append(f"Corrected needs_human {label.get('needs_human')} -> {target_nh}")
            label["needs_human"] = target_nh

        # Record fixes
        if fixes:
            log_entries.append((row_id, "fix", "; ".join(fixes)))

        # Update row with clean JSON
        r["messages"][2]["content"] = json.dumps({k: label[k] for k in KEYS})
        cleaned_rows.append(r)

    # Write cleaned training set
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for r in cleaned_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write cleaning log CSV
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action", "reason"])
        for entry in log_entries:
            writer.writerow(entry)

    print("Cleaning complete!")
    print(f"- Original rows: {len(rows)}")
    print(f"- Cleaned rows:  {len(cleaned_rows)}")
    print(f"- Total actions logged: {len(log_entries)}")
    actions = Counter(e[1] for e in log_entries)
    print(f"- Breakdown: {dict(actions)}")

if __name__ == "__main__":
    audit_and_clean()
