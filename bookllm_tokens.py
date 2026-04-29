#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Prepares data for the bookllm project
@author: Clement Gorin
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import fitz
import math
import pandas as pd
import re
import torch
import tqdm

from rouge_score import rouge_scorer
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from bookllm_utilities import *

# Parameters
backbone = 'meta-llama/Meta-Llama-3-70B' #! Base model
# backbone = 'meta-llama/Meta-Llama-3-8B' #! For testing
device   = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

#%% UTILITIES

def clean_text(text):
    text = re.sub(r'-\n', '',  text)  # Dehyphenation
    text = re.sub(r'\n+', ' ', text)  # Newlines to space
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    text = re.sub(r'[«»]', '"', text) # Page numbers
    text = re.sub(r'\\', '', text)    # Escaped characters
    return text.strip()

class TokenLoader:

    def __init__(self, tokens:torch.Tensor, n_context:int=256, n_target:int=32, stride:int=1, sample_size:int=None, batch_size:int=1, seed:int=0) -> None:
        self.tokens     = tokens
        self.n_context  = n_context
        self.n_target   = n_target
        self.stride     = stride
        self.batch_size = batch_size
        self.indices    = range(n_context, len(tokens) - n_target + 1, self.stride)
        if sample_size is not None:
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(list(self.indices), size=sample_size, replace=False)

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __iter__(self):
        batch_context, batch_target = [], []
        for i in self.indices:
            context = self.tokens[i - self.n_context:i]
            target  = self.tokens[i:i + self.n_target]
            batch_context.append(context)
            batch_target.append(target)
            if len(batch_context) == self.batch_size:
                yield torch.stack(batch_context), torch.stack(batch_target)  # (B, n_context), (B, n_target)
                batch_context, batch_target = [], []
        if batch_context:
            yield torch.stack(batch_context), torch.stack(batch_target)

#%% LOADS MODEL AND TOKENIZER

if 'model' not in dir():
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model  = AutoModelForCausalLM.from_pretrained(
        backbone, 
        quantization_config=config,
        device_map='auto',
        token='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')

if 'tokenizer' not in dir():
    tokenizer = AutoTokenizer.from_pretrained(
        backbone,
        token='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')
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = 'left'

#%% FORMATS DATA

books = search_data(f'{paths.data}', pattern='pdf$', kind='file')

for book in books:

    # Skips existing
    dstfile = f'{paths.results}/results_{filename(book)}_token_results.csv'
    if os.path.exists(dstfile):
        print(f'{book} already exists, skipping.')
        continue

    # Loads text
    pages = fitz.open(books)
    pages = [page for page in pages if page.get_text().strip()]

    # Extracts sentences
    tokens = []
    for page in pages:
        text = page.get_text()
        text = clean_text(text)
        text = tokenizer([text], return_tensors='pt').input_ids
        tokens.append(text)
    tokens = torch.cat(tokens, dim=1).squeeze(0)
    del page, text

    # Initialises loader
    loader = TokenLoader(tokens, n_context=256, n_target=32, stride=32, batch_size=1, batch_size=1, sample_size=None)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    # Computes metrics
    results  = []
    progress = tqdm.tqdm(loader, total=len(loader), desc='Verbatim test')
    running_rougel, running_perplex, running_n = 0, 0, 0
    for batch in progress:
        context, target = batch[0]
        # Verbatim completion
        inputs_c = tokenizer(context, return_tensors='pt').to(model.device)
        # max_new  = len(target.split()) + 5
        max_new  = len(tokenizer(target).input_ids) + 5
        with torch.no_grad():
            outputs = model.generate(**inputs_c, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        prediction   = tokenizer.decode(outputs[0, inputs_c.input_ids.shape[-1]:], skip_special_tokens=True)
        rougel_score = scorer.score(target, prediction)['rougeL'].fmeasure
        # Perplexity score
        inputs_f = tokenizer(context + ' ' + target, return_tensors='pt').to(model.device)
        labels   = inputs_f.input_ids.clone()
        labels[:, :inputs_c.input_ids.shape[-1]] = -100
        with torch.no_grad():
            perplex_score = torch.exp(model(**inputs_f, labels=labels).loss).item()
        # Statistics
        running_rougel  += rougel_score
        running_perplex += perplex_score
        running_n       += 1
        results.append({'context':context, 'target':target, 'prediction':prediction, 'rougel':rougel_score, 'perplexity':perplex_score})
        progress.set_postfix({'rougel': f'{running_rougel/running_n:.3f}', 'ppl': f'{running_perplex/running_n:.3f}'})

    # Saves results
    results = pd.DataFrame(results)
    results.to_csv(dstfile, index=False)

#%% 