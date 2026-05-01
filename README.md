# BookLLM — Detecting Book Memorisation in Language Models

A research toolkit for probing whether large language models have memorised copyrighted book content. The pipeline extracts text from PDFs, cleans it, then applies memorisation detection methods using a local LLM backbone.

## Overview

1. **Text extraction** — Load and clean PDF pages; split into sentences or tokenise
2. **Verbatim completion** — Check whether the model reproduces target text given preceding context
3. **Perplexity** — Measure model surprise on target text via masked loss
4. **Named entity recognition** — Compare entities in target vs. predicted text (sentence-level only)

## Repository Structure

```
code/
├── bookllm_utilities.py   # Shared helpers: paths, file search, folder management
├── bookllm_sentences.py   # Sentence-level metrics: verbatim completion, perplexity, NER
└── bookllm_tokens.py      # Token-level metrics: verbatim completion, perplexity
```

## Data

The pipeline expects PDF files under the data directory configured in `bookllm_utilities.py`:

```
bookllm/
└── galimard/
    └── 07-9782070323517.pdf   # Target book (PDF)
```

## Pipeline Execution

The scripts are designed to be run interactively (cell by cell) or as standalone modules:

```bash
# Run sentence-level memorisation probes
python bookllm_sentences.py

# Run token-level memorisation probes
python bookllm_tokens.py
```

Results are saved as CSV files under `paths.results` with filenames of the form `results_{book}_{label}.csv`.

## Key Methods

### Verbatim completion
Given preceding context, the model generates greedily up to the length of the target. ROUGE-L F-measure between the target and the prediction is computed as the similarity score.

- **Sentence-level** (`bookllm_sentences.py`): context is `n_context` preceding sentences; target is `n_target` sentences.
- **Token-level** (`bookllm_tokens.py`): context is `n_context` tokens; target is `n_target` tokens, using a sliding window with configurable stride.

### Perplexity
Context tokens are masked (`-100`) so that loss is computed only over target tokens. The masked loss is exponentiated to perplexity. Lower perplexity on a target passage relative to a control indicates higher model familiarity.

### Named entity recognition (sentence-level only)
A CamemBERT NER model extracts named entities from both the target and the predicted text. Entity lists are stored alongside ROUGE-L and perplexity in the output CSV.

## Dependencies

```
torch
transformers
pymupdf       # fitz
rouge_score
pandas
tqdm
scipy
matplotlib
```

Install with:

```bash
pip install torch transformers pymupdf rouge_score pandas tqdm scipy matplotlib
```

## Environment

`bookllm_utilities.py` sets data paths based on the system username:

| Username  | Environment       | Data path                    |
|-----------|-------------------|------------------------------|
| `root`    | Local laptop      | `~/Desktop/temporary/bookllm` |
| `clgorin` | Remote server     | `E:/ClementGorin/bookllm`    |

The model device is selected automatically: MPS → CUDA → CPU.

## Backbones

| Script                  | Variable          | Default model                              |
|-------------------------|-------------------|--------------------------------------------|
| `bookllm_sentences.py`  | `llama_backbone`  | `meta-llama/Meta-Llama-3-70B`              |
| `bookllm_sentences.py`  | `ner_backbone`    | `Jean-Baptiste/camembert-ner-with-dates`   |
| `bookllm_tokens.py`     | `backbone`        | `meta-llama/Meta-Llama-3-70B`              |

Both Llama models are loaded in 4-bit quantisation (`bfloat16`) via `BitsAndBytesConfig`.

## References

- Shi, W. et al. (2024). *Detecting Pretraining Data from Large Language Models*. ICLR 2024.
