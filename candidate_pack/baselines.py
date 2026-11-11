# baselines.py
# Evaluate zero-shot and in-context rule baselines on dev.jsonl

import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

FIXED_SYSTEM_PROMPT = "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."

RULES_SYSTEM_PROMPT = """You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON.
Strictly output a single JSON object with exactly these 8 keys:
- category: one of "payment_failed", "refund", "fraud", "account_access", "kyc", "offers", "other". "fraud" overrides all if customer mentions unauthorized transactions.
- priority: "P1", "P2", "P3". P1 if fraud, amount >= 50,000 INR (5,000,000 paise), or legal threats (RBI, ombudsman, consumer court, lawyer, legal notice). P2 if payment_failed/refund and amount >= 5,000 INR (500,000 paise), or account_access. Otherwise P3.
- amount_paise: integer in paise (1 INR = 100 paise, 1.3k = 130000, 1.5 lakh = 15000000). Only transaction/dispute amount, ignore balances or limits. null for account_access, kyc, other.
- txn_date: YYYY-MM-DD. Resolve relative dates against Received date. Ignore dates in quoted lines starting with >. null for account_access, kyc, other.
- txn_id: NP followed by 10 digits (e.g. NP1234567890). Normalize uppercase without spaces/symbols. Ignore non-NP reference numbers. null for account_access, kyc, other.
- channel: "upi", "card", "netbanking", "wallet" if explicitly stated, else null. null for account_access, kyc, other.
- language: "hi-en" if Hinglish (Hindi in Latin script mixed with English), "en" otherwise.
- needs_human: boolean true if priority is P1 or customer mentions repeated contact ("again and again", "third time", "baar baar", "already contacted"), else false.
Output ONLY the JSON object. Do not include markdown codeblocks or explanation."""

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def run_eval(model, tokenizer, rows, prompt_text, out_file):
    results = []
    print(f"Generating predictions for {len(rows)} tickets -> {out_file}")
    
    for item in tqdm(rows):
        uid = item["id"]
        ticket = item["messages"][1]["content"]
        
        chat = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": ticket}
        ]
        
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
            
        gen_tokens = gen[0][inputs["input_ids"].shape[1]:]
        out = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        results.append({"id": uid, "output": out})
        
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Done. Wrote {len(results)} outputs.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dev_file", default="candidate_pack/data/dev.jsonl")
    parser.add_argument("--mode", default="both", choices=["zero_shot", "rules", "both"])
    args = parser.parse_args()

    print(f"Loading {args.model} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto"
    )

    dev_data = load_data(args.dev_file)

    if args.mode in ["zero_shot", "both"]:
        print("\n=== Running Baseline 1: Zero-shot (Short Prompt) ===")
        run_eval(model, tokenizer, dev_data, FIXED_SYSTEM_PROMPT, "candidate_pack/predictions_baseline_zero_shot.jsonl")

    if args.mode in ["rules", "both"]:
        print("\n=== Running Baseline 2: In-Context Rules ===")
        run_eval(model, tokenizer, dev_data, RULES_SYSTEM_PROMPT, "candidate_pack/predictions_baseline_rules.jsonl")

if __name__ == "__main__":
    main()
