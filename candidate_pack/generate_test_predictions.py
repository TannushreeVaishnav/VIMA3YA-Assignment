# generate_test_predictions.py
# Generate test predictions for candidate_pack/data/test_inputs.jsonl

import json
import re
from datetime import datetime, timedelta

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]

LEGAL_RE = re.compile(r"\b(rbi|ombudsman|consumer\s+court|lawyer|legal\s+notice|legal\s+action)\b", re.I)
REPEAT_RE = re.compile(r"\b(third\s+time|again\s+and\s+again|baar\s+baar|already\s+contacted|multiple\s+times|several\s+times|contacted\s+before|second\s+time|4th\s+time|fourth\s+time|pehle\s+bhi|already\s+written)\b", re.I)
TXN_ID_RE = re.compile(r"\bnp\s*([0-9\s\-]{10,14})\b", re.I)
HINGLISH_WORDS = re.compile(r"\b(kya|hai|hain|nahi|nahin|maine|mera|meri|mere|kiya|gaya|gayi|bhai|kripya|baar|pehle|karo|raha|rahi|paise|rupaye|aaya|hua|huye|kyun|kab|kaise|madad|chahiye|karein|samarthan)\b", re.I)

MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12
}

def parse_incident_date(text, received_str):
    """Extract incident date and resolve relative dates against received date."""
    rec_date = datetime.strptime(received_str, "%Y-%m-%d")
    
    # Strip quoted email lines starting with >
    non_quoted = [l for l in text.split("\n") if not l.strip().startswith(">")]
    body = "\n".join(non_quoted)
    
    # Relative terms
    if re.search(r"\b(today|aaj)\b", body, re.I):
        return rec_date.strftime("%Y-%m-%d")
    if re.search(r"\b(yesterday|kal(?!\s+karo)|last\s+night)\b", body, re.I):
        return (rec_date - timedelta(days=1)).strftime("%Y-%m-%d")
    if re.search(r"\b(day\s+before\s+yesterday|parso)\b", body, re.I):
        return (rec_date - timedelta(days=2)).strftime("%Y-%m-%d")
    m_days = re.search(r"\b(\d+)\s+days?\s+ago\b", body, re.I) or re.search(r"\b(\d+)\s+din\s+pehle\b", body, re.I)
    if m_days:
        days = int(m_days.group(1))
        return (rec_date - timedelta(days=days)).strftime("%Y-%m-%d")

    # Explicit ISO date
    m_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", body)
    if m_iso:
        return m_iso.group(0)

    # DD/MM/YYYY
    m_dmy = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", body)
    if m_dmy:
        d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        try:
            return datetime(y, m, d).strftime("%Y-%m-%d")
        except:
            pass

    # DD Month (e.g. 15 Aug)
    m_dmonth = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-zA-Z]+)(?:\s+(\d{4}))?\b", body)
    if m_dmonth:
        d = int(m_dmonth.group(1))
        m_str = m_dmonth.group(2).lower()
        if m_str in MONTH_MAP:
            m = MONTH_MAP[m_str]
            y = int(m_dmonth.group(3)) if m_dmonth.group(3) else rec_date.year
            try:
                dt = datetime(y, m, d)
                if dt > rec_date:
                    dt = datetime(y - 1, m, d)
                return dt.strftime("%Y-%m-%d")
            except:
                pass

    return None

def parse_amount(text):
    """Extract dispute amount in paise, filtering out limits and balances."""
    non_quoted = [l for l in text.split("\n") if not l.strip().startswith(">")]
    body = "\n".join(non_quoted)
    
    # 1. Lakh shorthand
    m_lakh = re.search(r"(\d+(?:\.\d+)?)\s*lakh", body, re.I)
    if m_lakh:
        return int(round(float(m_lakh.group(1)) * 100000 * 100))
        
    # 2. k shorthand
    m_k = re.search(r"(\d+(?:\.\d+)?)\s*k\b", body, re.I)
    if m_k:
        prefix = body[max(0, m_k.start() - 20):m_k.start()].lower()
        if "version" not in prefix:
            return int(round(float(m_k.group(1)) * 1000 * 100))

    # 3. Currency patterns
    patterns = [
        r"(?:Rs\.?|₹|INR)\s*([\d,]+(?:\.\d{1,2})?)",
        r"([\d,]+(?:\.\d{1,2})?)\s*(?:rupees|paise|INR|Rs\.?|₹)"
    ]
    
    candidates = []
    for pat in patterns:
        for m in re.finditer(pat, body, re.I):
            val_str = m.group(1).replace(",", "")
            try:
                val = float(val_str)
                ctx = body[max(0, m.start() - 35):min(len(body), m.end() + 35)].lower()
                is_ignored = any(w in ctx for w in ["balance", "daily limit", "limit is", "limit of", "fee", "fees", "charge"])
                candidates.append((m.start(), val, is_ignored))
            except:
                pass
                
    valid = [c for c in candidates if not c[2] and c[1] > 0]
    if valid:
        return int(round(valid[0][1] * 100))
        
    return None

def parse_txn_id(text):
    """Extract NP + 10 digits transaction ID."""
    non_quoted = [l for l in text.split("\n") if not l.strip().startswith(">")]
    body = "\n".join(non_quoted)
    m = TXN_ID_RE.search(body)
    if m:
        digits = re.sub(r"[^0-9]", "", m.group(1))
        if len(digits) == 10:
            return f"NP{digits}"
    return None

def parse_channel(text):
    """Extract payment channel if stated explicitly."""
    non_quoted = [l for l in text.split("\n") if not l.strip().startswith(">")]
    body = "\n".join(non_quoted)
    
    if re.search(r"\b(netbanking|internet\s+banking|net\s+banking|online\s+banking)\b", body, re.I):
        return "netbanking"
    if re.search(r"\b(credit\s+card|debit\s+card|card\s+payment|visa|mastercard|new\s+card|my\s+card)\b", body, re.I):
        return "card"
    if re.search(r"\b(upi|gpay|google\s+pay|phonepe)\b", body, re.I):
        return "upi"
    if re.search(r"\b(wallet|nimbuspay\s+wallet)\b", body, re.I):
        return "wallet"
    return None

def classify_ticket(user_text):
    """Predict triage fields according to SCHEMA.md rules."""
    non_quoted = [l for l in user_text.split("\n") if not l.strip().startswith(">")]
    body = "\n".join(non_quoted)
    
    # Received date
    m_rec = re.search(r"Received:\s*(\d{4}-\d{2}-\d{2})", user_text)
    rec_date_str = m_rec.group(1) if m_rec else "2026-08-01"
    
    # Language detection
    hinglish_matches = HINGLISH_WORDS.findall(body)
    is_hinglish = len(hinglish_matches) >= 2 or any(w in body.lower() for w in ["kya", "nahi", "maine", "paise kaat", "help karo", "sir ji"])
    language = "hi-en" if is_hinglish else "en"
    
    # Category detection rules
    fraud_re = re.compile(r"(didn't\s+make|did\s+not\s+make|not\s+me|unauthorized|unrecognised|unrecognized|stolen|scam|fraud|maine\s+nahi\s+kiya|kisi\s+ne\s+otp|otp\s+maang\s+ke|unapproved|compromised)", re.I)
    account_re = re.compile(r"(blocked\s+account|locked\s+account|account\s+locked|blocked\s+pin|forgot\s+passcode|forgot\s+password|reset\s+password|otp\s+not\s+received|can't\s+log\s*in|cannot\s+log\s*in|login\s+issue|pin\s+number)", re.I)
    kyc_re = re.compile(r"(kyc|identity\s+verification|verify\s+my\s+identity|prove\s+my\s+identity|aadhaar|pan\s+card|video\s+kyc|documents\s+pending|verification\s+failed|verification\s+pending|document\s+upload)", re.I)
    offers_re = re.compile(r"(cashback|promo\s+code|coupon|scratch\s+card|referral\s+bonus|reward\s+points|offers|reawrd)", re.I)
    refund_re = re.compile(r"(refund|cancelled\s+order|return|charged\s+twice|charged\s+more\s+than\s+once|double\s+charge|money\s+back|paise\s+wapas)", re.I)
    payment_failed_re = re.compile(r"(failed|declined|not\s+go\s+through|didn't\s+go\s+through|didn't\s+work|stuck|pending\s+transaction|did\s+not\s+reach|decline|paise\s+kaat\s+liye)", re.I)
    
    if fraud_re.search(body):
        category = "fraud"
    elif account_re.search(body):
        category = "account_access"
    elif kyc_re.search(body):
        category = "kyc"
    elif offers_re.search(body):
        category = "offers"
    elif refund_re.search(body):
        category = "refund"
    elif payment_failed_re.search(body):
        category = "payment_failed"
    else:
        category = "other"
        
    # Transaction fields: null for account_access, kyc, other
    if category in {"account_access", "kyc", "other"}:
        amount_paise = None
        txn_date = None
        txn_id = None
        channel = None
    else:
        amount_paise = parse_amount(body)
        txn_date = parse_incident_date(body, rec_date_str)
        txn_id = parse_txn_id(body)
        channel = parse_channel(body)
        
    # Priority hierarchy
    is_legal = bool(LEGAL_RE.search(body))
    is_fraud = (category == "fraud")
    is_large = (amount_paise is not None and amount_paise >= 5000000)
    
    if is_fraud or is_large or is_legal:
        priority = "P1"
    else:
        is_p2 = ((category in {"payment_failed", "refund"} and amount_paise is not None and amount_paise >= 500000) or (category == "account_access"))
        priority = "P2" if is_p2 else "P3"
            
    # needs_human rule
    is_repeat = bool(REPEAT_RE.search(body))
    needs_human = True if (priority == "P1" or is_repeat) else False
    
    return {
        "category": category,
        "priority": priority,
        "amount_paise": amount_paise,
        "txn_date": txn_date,
        "txn_id": txn_id,
        "channel": channel,
        "language": language,
        "needs_human": needs_human
    }

def main():
    test_file = "candidate_pack/data/test_inputs.jsonl"
    out_file = "predictions_test.jsonl"
    
    with open(test_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
        
    print(f"Generating predictions for {len(rows)} test tickets...")
    predictions = []
    
    for r in rows:
        rid = r["id"]
        ticket_text = r["messages"][1]["content"]
        record = classify_ticket(ticket_text)
        predictions.append({"id": rid, "output": json.dumps(record)})
        
    with open(out_file, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Also keep a copy inside candidate_pack
    with open("candidate_pack/predictions_test.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(predictions)} predictions -> {out_file}")

if __name__ == "__main__":
    main()
