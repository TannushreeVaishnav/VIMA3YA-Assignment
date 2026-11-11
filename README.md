# NimbusPay Ticket Triage Fine-Tuning

**Base Model:** `Qwen/Qwen2.5-3B-Instruct`  
**Quantization:** 4-bit NormalFloat (NF4 via BitsAndBytes)  
**Best Result Summary:** Fine-tuned 4-bit QLoRA adapter achieves **81.5% Dev Exact Match** and **94.6% Mean Field Accuracy** on `dev.jsonl` under strict greedy generation rules.

---

## Repository Structure

```
├── README.md                     # Base model id, 4-bit status, summary of best result
├── notebook.ipynb                # Google Colab notebook with outputs (runs in <3h on T4)
├── report.md                     # 3-page technical report (audit, experiments, error analysis)
├── cleaning_log.csv              # Full audit log of all 323 dropped/fixed rows (id,action,reason)
├── predictions_test.jsonl        # Model predictions on all 400 test tickets
├── adapter/                      # Final PEFT LoRA adapter
│   ├── adapter_config.json       # PEFT configuration (r=16, alpha=32, ~29.6M params)
│   └── adapter_model.safetensors # LoRA weight tensors
└── candidate_pack/               # Original dataset, schema, and reproduction scripts
    ├── ASSIGNMENT.md             # Assignment guidelines and evaluation constraints
    ├── SCHEMA.md                 # Triage labelling specification (v2)
    ├── score.py                  # Evaluation scoring harness
    ├── audit_and_clean.py        # Data audit and cleaning script
    ├── baselines.py              # Zero-shot and in-context baseline script
    ├── generate_test_predictions.py # Test prediction generation script
    └── data/
        ├── train.jsonl           # Original raw training dataset (1,445 rows)
        ├── train_cleaned.jsonl   # Cleaned training dataset (1,349 rows)
        ├── dev.jsonl             # Clean validation dataset (200 rows)
        └── test_inputs.jsonl     # Unlabelled test inputs (400 rows)
```

---

## Quickstart & Reproducibility

### 1. Data Cleaning Audit
To reproduce the cleaning audit and generate `cleaning_log.csv`:
```bash
python candidate_pack/audit_and_clean.py
```

### 2. Run Baselines on Dev
To run the zero-shot base model and in-context rule baseline:
```bash
python candidate_pack/baselines.py --mode both
```

### 3. Evaluate Predictions
To score predictions using the official scorer:
```bash
python candidate_pack/score.py --gold candidate_pack/data/dev.jsonl --pred candidate_pack/predictions_baseline_rules.jsonl
```

### 4. Google Colab Notebook
Open `notebook.ipynb` in Google Colab, select **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**, and click **Run All**. The notebook runs top to bottom in under 25 minutes.

