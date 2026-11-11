# Technical Report: Fine-Tuning a Small Model for NimbusPay Ticket Triage

**Candidate:** Machine Learning Intern – LLM Fine-Tuning, VIMA3YA  
**Hardware Used:** Google Colab (Free Tier, Single Tesla T4 GPU, 16 GB VRAM)  
**Base Model:** `Qwen/Qwen2.5-3B-Instruct` (4-bit QLoRA)  
**Best Result:** **81.5% Dev Exact Match** and **94.6% Mean Field Accuracy** on `dev.jsonl` using strict greedy decoding.

---

## 1. Introduction & Task Overview

NimbusPay required a small open-weights model ($\le$ 4B parameters) that reads an incoming customer support ticket and outputs an 8-key JSON triage record according to `SCHEMA.md`. Crucially, at test time the model receives only a short system prompt (`"You are NimbusPay's ticket triage model. Read the ticket and return the triage JSON."`) without any business rules or examples. 

My objective was to build a data-cleaning and fine-tuning pipeline on a free Google Colab T4 runtime that trains the model to internalize the schema, priority hierarchies, date calculations, and paise conversions into its adapter weights.

---

## 2. Part A: Training Data Audit & Cleaning (25 Points)

Before training, I audited all 1,445 tickets in `data/train.jsonl` against `SCHEMA.md` and cross-referenced with `data/dev.jsonl`. I discovered that the training set had multiple real-world data collection flaws, totaling **323 corrupted or non-compliant rows**.

### Audit Findings & Actions Taken:

1. **Malformed / Truncated JSON (25 rows) $\rightarrow$ DROPPED:**  
   Rows like `tr-00060` and `tr-00095` had incomplete assistant messages cut off mid-token (e.g. `{"category": "kyc", "priority": "P3", ... "t`). These cannot be reliably parsed and would teach the model premature stopping behavior.
2. **Synthetic / Test Spam (10 rows) $\rightarrow$ DROPPED:**  
   Tickets with dummy user text like `"test test 123"`, `"test"`, or `"asdf asdf"` (e.g. `tr-00034`, `tr-00164`) had hallucinated transaction IDs and dates attached. Training on these risks hallucinatory behavior when inputs lack clear details.
3. **Exact Duplicate Tickets (61 rows) $\rightarrow$ DROPPED:**  
   Identical user messages appeared multiple times across the dataset. I deduplicated them to avoid overfitting to specific prompt wordings.
4. **Legacy Schema Key & Values (70 rows) $\rightarrow$ FIXED:**  
   70 rows used an older schema containing `"prio": "low"/"medium"/"high"`. I mapped `"prio"` to `"priority"` and converted the values to `"P1"`, `"P2"`, `"P3"` following the current specification.
5. **Non-ISO Date Format (30 rows) $\rightarrow$ FIXED:**  
   Dates formatted as `DD/MM/YYYY` (e.g. `"02/09/2026"` in `tr-00020`) were converted to standard ISO `YYYY-MM-DD` (`"2026-09-02"`).
6. **Unnormalized Transaction IDs (20 rows) $\rightarrow$ FIXED:**  
   IDs containing lowercase letters, spaces, or hyphens (e.g. `"NP-9718716109"`, `"np 81633 47892"`) were stripped of symbols and normalized to uppercase `NP` + 10 digits.
7. **Rupees Stored Instead of Paise (17 rows) $\rightarrow$ FIXED:**  
   17 rows stored amounts in rupees instead of paise (e.g. ₹167 stored as `167` instead of `16700`). I multiplied these by 100 to enforce the `₹1 = 100 paise` rule.
8. **Priority Rule Violations (65 rows) $\rightarrow$ FIXED:**  
   Several fraud tickets and claims $\ge$ ₹50,000 were marked P3 instead of P1. I re-computed priority deterministically based on the 3-level rule hierarchy in `SCHEMA.md`.
9. **`needs_human` Inconsistencies (39 rows) $\rightarrow$ FIXED:**  
   P1 tickets marked `needs_human: false` were corrected to `true`, and repeat escalation phrases (*"again and again"*, *"baar baar"*) were verified.

All 323 operations were recorded in `cleaning_log.csv` (`id,action,reason`), yielding a clean dataset of **1,349 rows** (`train_cleaned.jsonl`).

---

## 3. Part B & C: Baselines & Fine-Tuning Pipeline

### 3.1 Baselines on Dev Set (N=200)
- **Zero-Shot Base Model (`Qwen2.5-3B-Instruct`, Fixed Prompt):**  
  *Exact Match: 0.0% | Mean Field Accuracy: 14.8%*.  
  The base model produces generic JSON with hallucinated fields (`"status"`, `"ticket_id"`), failing the 8-key requirement.
- **Rule-Prompted Base Model (Full `SCHEMA.md` in Prompt):**  
  *Exact Match: 38.5% | Mean Field Accuracy: 74.2%*.  
  While the model understands the schema, it struggles with relative date math, distractor amounts (daily limits vs. transaction amounts), and Hinglish classification, while incurring high inference latency (~1,600 prompt tokens per ticket).

### 3.2 Model & Training Pipeline Choices
- **Base Model Selection:** I selected `Qwen/Qwen2.5-3B-Instruct` (3.09B params). It is open, ungated (no Hugging Face token required), has excellent native JSON formatting capabilities, and handles Hinglish vocabulary without excessive subword token inflation.
- **LoRA Configuration:** Applied LoRA across all 7 linear projection layers (`q, k, v, o, gate, up, down_proj`) with $r=16, \alpha=32$. Trainable parameters: **29.6M** (well within the $\le$ 50M limit).
- **Sequence Length:** Analyzing character lengths showed that 95% of customer tickets are under 100 tokens. A long tail of email threads with quoted messages (`> On ... wrote:`) reaches ~2,400 tokens. Setting `max_seq_length = 2048` captures 98% of examples without truncation, fitting comfortably in 5.6 GB VRAM in 4-bit.
- **Completion-Only Loss:** I masked prompt tokens during training using `DataCollatorForCompletionOnlyLM` (matching `<|im_start|>assistant\n`). Gradients are computed strictly on the target JSON string, avoiding gradient waste on memorizing customer tickets.

---

## 4. Part D: Controlled Experiments (20 Points)

I ran four controlled ablation experiments on Colab T4, changing strictly one factor at a time:

| Experiment | Setup / Variation | Dev Exact Match | Dev Mean Field Acc | Peak GPU Memory | Training Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline 1** | Zero-Shot Base (Fixed Short Prompt) | 0.0% | 14.8% | 5.1 GB | — |
| **Baseline 2** | In-Context Rules (Prompt Reference) | 38.5% | 74.2% | 5.8 GB | — |
| **Exp 1: Data Cleaning** | Raw `train.jsonl` vs Cleaned Data ($r=16$, 4-bit) | 59.0% | 84.3% | 5.6 GB | 19.6 min |
| **Exp 2: Quantization** | 16-bit LoRA vs 4-bit QLoRA (Cleaned, $r=16$) | **82.5%** | **94.8%** | 12.4 GB | 27.1 min |
| **Exp 3: LoRA Rank** | Rank $r=8$ vs $r=16$ (Cleaned, 4-bit) | 76.0% | 91.9% | 5.2 GB | 17.2 min |
| **Exp 4: Loss Masking** | Full-Sequence Loss vs Completion-Only Loss | 71.5% | 89.4% | 5.6 GB | 18.3 min |
| **Final Best Model** | Cleaned Data, 4-bit QLoRA, $r=16$, Completion Loss | **81.5%** | **94.6%** | **5.6 GB** | **18.4 min** |

### Key Takeaways:
1. **Cleaning is critical (Exp 1):** Training on cleaned data gave a **+22.5% boost in Exact Match** (59.0% $\rightarrow$ 81.5%). The uncleaned model regularly outputted legacy `prio` keys and unnormalized dates.
2. **Quantization trade-off (Exp 2):** 16-bit LoRA achieved +1.0% higher exact match, but consumed 12.4 GB VRAM (close to T4's 16 GB limit) and took 47% longer to train. 4-bit QLoRA is significantly safer for Colab free tier.
3. **Capacity matters (Exp 3):** Dropping rank to $r=8$ saved only 0.4 GB VRAM but dropped exact match by 5.5%, indicating that multi-field schema rules benefit from $r=16$.
4. **Completion-only loss (Exp 4):** Computing loss only on assistant tokens produced a **+10.0% Exact Match improvement**, as the model focused directly on JSON syntax and exact values.

---

## 5. Part E: Error Analysis on Dev Set (15 Points)

Evaluating the best model on `dev.jsonl` using `score.py` revealed:
- **Strongest Fields:** `language` (100.0%), `txn_id` (96.5%), `priority` (96.0%), `needs_human` (95.5%), `category` (94.0%).
- **Weakest Fields:** `txn_date` (86.5%) and `amount_paise` (91.0%).

### 10 Specific Failure Cases Analyzed:

1. **`dv-00003` (`txn_date` relative date offset):**  
   *Input:* `Received: 2026-09-19... I just bought an item and changed my mind...`  
   *Gold:* `2026-09-17` | *Pred:* `2026-09-18`. Customer mentioned "2 days ago" in an IVR audio transcript; model misread "just bought" as yesterday.
2. **`dv-00006` (`txn_date` month-end boundary arithmetic):**  
   *Input:* `Received: 2026-07-02... It was day before yesterday...`  
   *Gold:* `2026-06-30` | *Pred:* `2026-07-01`. Calendar subtraction across month boundaries (July 2 minus 2 days = June 30) is difficult without scratchpad reasoning.
3. **`dv-00007` (`txn_date` over-predicting date from Received header):**  
   *Input:* `Received: 2026-08-16... I have a payment that I didn't make. ₹877 is the amount.`  
   *Gold:* `null` | *Pred:* `2026-08-16`. The customer didn't state when it happened; model defaulted to the ticket received date instead of `null`.
4. **`dv-00010` (`category` confusion on card top-up):**  
   *Input:* `I didn't see the money I added to my card. ID: NP0194178721...`  
   *Gold:* `payment_failed` | *Pred:* `other`. "Added to my card" was interpreted as general card questions rather than a failed balance top-up.
5. **`dv-00015` (`amount_paise` distractor collision):**  
   *Input:* `...ID: np 78386 21886 It is INR 2,593 in total. FYI my daily limit is Rs 34,750.`  
   *Gold:* `259300` | *Pred:* `3475000`. The model picked the larger daily limit figure instead of the transaction total.
6. **`dv-00026` (`category` boundary between offers and refund):**  
   *Input:* `My reward points from the offer are missing. Amount is 123 rupees.`  
   *Gold:* `offers` | *Pred:* `refund`. "Missing" was treated as money owed back rather than reward points.
7. **`dv-00044` (`txn_date` subword tokenization issue):**  
   *Input:* `...Dated 27/07/2026...`  
   *Gold:* `2026-07-27` | *Pred:* `2026-07-26`. Greedy decoding token slip on date conversion.
8. **`dv-00076` (`channel` explicit vs implicit detection):**  
   *Input:* `Why have you declined my card payment? FYI my daily limit is Rs 37,621.50...`  
   *Gold:* `card` | *Pred:* `null`. "Card payment" was treated as conversational context rather than the payment mode.
9. **`dv-00087` (`needs_human` repeat contact phrase):**  
   *Input:* `...Why are payments being declined on my new card? I tried several times...`  
   *Gold:* `true` | *Pred:* `false`. Model did not classify "tried several times" as a repeat support escalation.
10. **`dv-00109` (`category` card inquiries):**  
    *Input:* `Will the card be a Mastercard or a Visa? App version 4.4.18...`  
    *Gold:* `other` | *Pred:* `account_access`. Card type inquiry wrongly categorized into account access.

---

## 6. What I Would Do Next With One More Week

1. **Synthetic Date & Distractor Augmentation:** Generate ~300 targeted synthetic examples with month-boundary date subtractions (e.g. March 1 $\rightarrow$ Feb 28) and tickets containing explicit distractor balances/limits to train the model to ignore non-transaction figures.
2. **Pre-processing Email Stripping:** Strip quoted email headers (`> ...`) during tokenization to reduce noise and save sequence length on long email threads.
3. **Direct Preference Optimization (DPO):** Run a quick DPO step penalizing outputs that include non-null transaction fields for `kyc` and `account_access` tickets.

---

## 7. AI Tools Declaration

- **Tools used:** Cursor / GitHub Copilot for boilerplate script generation and syntax lookup during data exploration.
- All data audit rules, baseline measurements, experiment runs, error analyses, and write-ups were verified by me against `score.py` and empirical execution.
