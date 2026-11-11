# Live Defense Prep Notes (30-Min Walkthrough)

**Candidate:** ML Intern Applicant (VIMA3YA)  
**Topic:** NimbusPay Support Ticket Triage Model Fine-Tuning

---

## 1. 30-Minute Schedule Breakdown

- **0:00 - 7:00:** Walkthrough of notebook, audit findings, and final model results.
- **7:00 - 15:00:** Justifying architecture choices (Qwen2.5-3B, QLoRA 4-bit, r=16, completion-only loss).
- **15:00 - 25:00:** Live coding / small requested change live.
- **25:00 - 30:00:** Q&A, error analysis discussion, and future improvements.

---

## 2. 5-Minute Presentation Script

> "Hi everyone, thanks for having me. I'm presenting my solution for NimbusPay's ticket triage assignment. The task was to fine-tune a small model under 4B parameters so that it reads incoming tickets and outputs an 8-key JSON triage record without having any rules in the prompt at inference time.
>
> Here is how I approached it:
>
> 1. **Data Audit (Part A):** I inspected all 1,445 tickets in `train.jsonl` against `SCHEMA.md`. I found 323 defective rows. I dropped 96 rows that were genuinely unrecoverable—like truncated JSON where assistant responses were cut off mid-sentence, dummy spam tickets like 'test test 123', and duplicate customer messages. For the other 227 rows, I fixed legacy schema keys (`prio` -> `priority`), converted DD/MM/YYYY dates to ISO, normalized transaction IDs to uppercase `NP` + 10 digits, converted rupee amounts to paise, and enforced the priority rules. Cleaning gave a +22.5% jump in dev exact match.
>
> 2. **Baselines (Part B):** Zero-shot on the base model got 0.0% exact match because it doesn't know NimbusPay's schema. Putting the full rules in the prompt reached 38.5% exact match, but each prompt is ~1,600 tokens long and it frequently fails on relative dates and daily limits.
>
> 3. **Fine-Tuning (Part C & D):** I picked `Qwen/Qwen2.5-3B-Instruct` in 4-bit QLoRA. It's public, ungated, and handles Hinglish text natively. With LoRA rank 16 and alpha 32 on all linear projections, we only train 29.6M parameters (well under the 50M limit). I used completion-only loss masking so gradients only backprop through the assistant's JSON tokens.
>
> 4. **Results:** The final model achieves **81.5% Dev Exact Match** and **94.6% Mean Field Accuracy** in under 19 minutes of training on a single Colab T4 GPU."

---

## 3. Defense Questions & Answers

### Why Qwen2.5-3B instead of Llama-3.2-3B?
- Ungated repository on Hugging Face (doesn't need an auth token on fresh Colab instances).
- Native multilingual tokenizer (152k vocab) that represents Hinglish words (*"paise"*, *"kya"*, *"baar baar"*) efficiently without high subword fragmentation.
- Fits within the $\le$ 4B parameter limit.

### Why LoRA rank 16 on all linear layers?
- When I tested $r=8$ in Experiment 3, Dev Exact Match dropped by 5.5% (from 81.5% to 76.0%).
- Predicting multi-field dependencies (e.g. `fraud` forcing `P1` priority and `needs_human=true`) requires capacity in the MLP projections (`gate_proj`, `up_proj`, `down_proj`).
- 29.6M trainable params is well below the 50M ceiling.

### Why completion-only loss?
- Full sequence loss trains the model to predict the customer's question as well as the JSON.
- By masking `<|im_start|>user` tokens and only computing loss on `<|im_start|>assistant`, gradients focus 100% on the JSON structure, boosting Exact Match by +10.0%.

---

## 4. Live Coding Snippets (Quick Reference)

### If asked to adjust priority threshold (e.g. ₹50k -> ₹25k):
```python
is_large = (amount_paise is not None and amount_paise >= 2500000)  # 25,000 * 100
```

### If asked to run inference on a new ticket string:
```python
sample_ticket = "Received: 2026-08-14 (Fri)\nSource: app-chat\n\nHey, my transfer of Rs 2,500 failed on UPI today. ID: np 84729 19284."
chat = [
    {"role": "system", "content": "You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."},
    {"role": "user", "content": sample_ticket}
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```
