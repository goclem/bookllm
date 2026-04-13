# BookLLM — Detecting Book Memorisation in Language Models

A research toolkit for probing whether large language models have memorised copyrighted book content. The pipeline extracts text from PDFs, cleans and sentences it, then applies three complementary memorisation detection methods using a local LLM backbone.

## Overview

1. **Text extraction** — Load and clean PDF pages; split into sentences
2. **Verbatim completion** — Check whether the model reproduces target sentences given preceding context
3. **Perplexity** — Measure model surprise on target text via sliding-window NLL
4. **Min-K% Prob** — Identify low-probability tokens to score memorisation (Shi et al. 2024)

## Repository Structure

```
code/
├── bookllm_utilities.py   # Shared helpers: paths, file search, folder management
└── bookllm_data.py        # Text extraction, cleaning, and memorisation probes
```

## Data

The pipeline expects a `bookllm/` directory under `~/Desktop/temporary/` with the following layout:

```
bookllm/
└── galimard/
    └── 07-9782070323517.pdf   # Target book (PDF)
```

## Pipeline Execution

The scripts are designed to be run interactively (cell by cell) or as standalone modules:

```bash
# Run memorisation probes on a book
python bookllm_data.py
```

Key outputs printed to stdout:
- Verbatim completion results (context / target / prediction / exact match / token overlap)
- Perplexity of target vs. control text
- Min-K% Prob scores for target vs. control passages

## Key Methods

### Verbatim completion
Given `n_context` preceding sentences as a prompt, the model generates up to `len(target) + 5` tokens greedily. Exact string match and token-level Jaccard overlap are reported.

### Perplexity (sliding window)
Token-level negative log-likelihoods are accumulated over non-overlapping chunks of `max_position_embeddings` length, then exponentiated to perplexity. Lower perplexity on a target passage relative to a control indicates higher familiarity.

### Min-K% Prob (Shi et al. 2024)
The bottom-K% of per-token log-probabilities are averaged. A more negative score suggests the model assigns low probability to specific tokens — a signal of *less* memorisation. Comparing target vs. control scores provides a relative memorisation estimate.

## Dependencies

```
torch
transformers
pymupdf       # fitz
```

Install with:

```bash
pip install torch transformers pymupdf
```

## Environment

`bookllm_utilities.py` sets data paths based on the system username, supporting both a local laptop (`root`) and a remote server (`clgorin`) environment. The model device is selected automatically: Apple MPS → CUDA → CPU.

## Backbone

Default model: `meta-llama/Meta-Llama-3-8B-Instruct` (loaded in `bfloat16` with `device_map='auto'`). Change the `backbone` variable in `bookllm_data.py` to use a different HuggingFace model.

## References

- Shi, W. et al. (2024). *Detecting Pretraining Data from Large Language Models*. ICLR 2024.
