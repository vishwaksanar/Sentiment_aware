# Sentiment-Aware Mental Health Chatbot

This repository contains a modular data pipeline for a sentiment-aware mental-health support chatbot. The pipeline prepares raw dialogue datasets, aligns them to a unified distress taxonomy, validates samples with an LLM, constructs DPO preference pairs, and supports DPO adapter training and inference comparison.

## Pipeline Order

```text
Raw JSONL datasets
-> raw dataset statistics
-> preprocessing / duplicate instruction removal
-> heuristic category alignment
-> LLM evaluation, category correction, and response refinement
-> DPO train/validation/test construction
-> optional DPO training
-> optional base vs DPO inference comparison
```

## Colab Setup

Use GPU in Colab:

```text
Runtime -> Change runtime type -> GPU
```

Clone the repository:

```python
%cd /content
!rm -rf Sentiment_aware
!git clone --depth 1 https://github.com/vishwaksanar/Sentiment_aware.git
%cd Sentiment_aware
!ls
```

Install training dependencies only if you plan to train or run local Llama inference:

```python
!pip install -q unsloth trl peft accelerate bitsandbytes transformers datasets
```

For the LLM evaluation stage, set a Groq/OpenAI-compatible Llama endpoint:

```python
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("Groq API key: ")
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_MODEL"] = "llama-3.1-8b-instant"
```

## Input Files

The examples below assume the raw datasets are available at:

```text
/content/counsellingED_cleaned.jsonl
/content/empathetic_dialogues_llama_core.jsonl
```

If the files are in Google Drive, mount Drive and replace paths:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Example Drive path:

```text
/content/drive/MyDrive/Raw dataset/Preprocessed/counsellingED_cleaned.jsonl
```

## 1. Raw Dataset Statistics

Run this before cleaning/alignment:

```python
!mkdir -p outputs

!python -m sentiment_aware.cli stats \
"Empathetic Dialogues=/content/empathetic_dialogues_llama_core.jsonl" \
"Counseling Data=/content/counsellingED_cleaned.jsonl" \
--output outputs/raw_dataset_stats.json \
--top-n 15
```

This reports sample counts, emotion-label counts, average user/response length, support-seeking percentage, safety-sensitive signal percentage, and advice-style response percentage.

## 2. Heuristic Alignment With Deduplication

The loader automatically removes the phase-3 wrapper:

```text
The user is feeling anxiety. Write an empathetic and supportive response.

User: ...
```

and keeps only the actual user message. The embedded emotion is preserved as the label.

Run counselling alignment with duplicate-instruction removal:

```python
!python -m sentiment_aware.cli align \
"/content/counsellingED_cleaned.jsonl" \
outputs/counsellingED_cleaned_aligned_deduped.jsonl \
--source counsellingED_cleaned \
--dedupe-instruction
```

Run empathetic-dialogues alignment if needed:

```python
!python -m sentiment_aware.cli align \
"/content/empathetic_dialogues_llama_core.jsonl" \
outputs/empathetic_dialogues_aligned_deduped.jsonl \
--source empathetic_dialogues_llama_core \
--dedupe-instruction
```

## 3. LLM Evaluation

The LLM evaluator:

- checks if the heuristic category is correct,
- writes `final_category`,
- checks if the response is safe and empathetic,
- writes `final_response`,
- preserves `original_category` and `original_response`.

For a quick test:

```python
!python scripts/run_llm_self_consistency_eval.py \
outputs/counsellingED_cleaned_aligned_deduped.jsonl \
outputs/counsellingED_llama_category_response_eval_5.jsonl \
--limit 5 \
--votes 1 \
--sleep 3
```

For self-consistency, increase votes:

```python
!python scripts/run_llm_self_consistency_eval.py \
outputs/counsellingED_cleaned_aligned_deduped.jsonl \
outputs/counsellingED_llama_category_response_eval_5.jsonl \
--limit 5 \
--votes 3 \
--sleep 10
```

Use smaller limits or longer sleep if the API returns rate-limit errors.

## 4. Build DPO Data From LLM Output

This uses:

```text
final_category  -> prompt category
final_response  -> chosen response
original_response -> rejected response if the LLM revised it
```

Run:

```python
!python -m sentiment_aware.cli build-dpo-from-llm \
outputs/counsellingED_llama_category_response_eval_5.jsonl \
outputs/dpo_from_llm_eval
```

Inspect:

```python
!ls outputs/dpo_from_llm_eval
!cat outputs/dpo_from_llm_eval/summary.json
!head -c 2000 outputs/dpo_from_llm_eval/train.json
```

## 5. Optional DPO Training

Training code is in:

```text
sentiment_aware/training.py
```

It uses Unsloth + TRL `DPOTrainer` with QLoRA:

```text
base model: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
LoRA rank: 4
LoRA alpha: 8
DPO beta: 0.1
```

Load the JSON split files with Hugging Face `datasets`, then call `train_dpo`.

```python
from datasets import load_dataset
from sentiment_aware.training import train_dpo

dataset = load_dataset(
    "json",
    data_files={
        "train": "outputs/dpo_from_llm_eval/train.json",
        "validation": "outputs/dpo_from_llm_eval/validation.json",
        "test": "outputs/dpo_from_llm_eval/test.json",
    },
)

trainer = train_dpo(dataset)
```

The adapter is saved to:

```text
outputs/llama3_2_3b_dpo_adapter
```

## 6. Optional Base vs DPO Inference

After training or uploading an adapter, compare base and DPO-aligned responses:

```python
!python -m sentiment_aware.cli compare-inference \
"i feel like i cannot handle this stress anymore" \
--category stress \
--adapter-path "/content/llama3_2_3b_dpo_adapter/content/llama3_2_3b_dpo_adapter"
```

Optional generation settings:

```text
--model-name "unsloth/Llama-3.2-3B-Instruct"
--max-new-tokens 200
--temperature 0.7
--top-p 0.9
```

## Local Development

Default local repo path:

```text
C:\Projects\Sentiment_aware
```

Run tests:

```powershell
python -m unittest tests.test_pipeline
```

Push updates:

```powershell
git add .
git commit -m "your message"
git push origin main
```
