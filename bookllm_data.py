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
# backbone = 'meta-llama/Meta-Llama-3-70B'
backbone = 'meta-llama/Meta-Llama-3-8B'
device   = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

#%% UTILITIES

def clean_text(text):
    text = re.sub(r'-\n', '',  text)  # Dehyphenation
    text = re.sub(r'\n+', ' ', text)  # Newlines to space
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces
    text = re.sub(r'[«»]', '"', text) # Page numbers
    text = re.sub(r'\\', '', text)    # Escaped characters
    return text.strip()

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

class SentenceLoader:

    def __init__(self, sentences:list, n_context:int=3, sample_size:int=None, seed:int=0) -> None:
        self.sentences  = sentences
        self.n_context  = n_context
        self.batch_size = 1
        self.indices    = range(n_context, len(sentences))
        if sample_size is not None:
            rng = np.random.default_rng(seed)
            self.indices = np.random.choice(self.indices, size=sample_size, replace=False)

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __iter__(self):
        batch = []
        for i in self.indices:
            context = ' '.join(self.sentences[max(0, i-self.n_context):i])
            target  = self.sentences[i]
            batch.append((context, target))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

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

for title in ...:

    # Loads dataé
    book  = fitz.open(f'{paths.data}/galimard/{title}.pdf')
    book  = [page for page in book if page.get_text().strip()]

    # Extracts sentences
    sentences = []
    for page in book:
        text = page.get_text()
        text = clean_text(text)
        text = split_sentences(text)
        sentences.extend(text)
    del page, text

    # Initialises loader
    loader = SentenceLoader(sentences, n_context=3, sample_size=100000, seed=0)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    # Computes metrics
    results  = []
    progress = tqdm.tqdm(loader, total=len(loader), desc='Verbatim test')
    running_rougel, running_perplex, running_n = 0, 0, 0
    for batch in progress:
        context, target = batch[0]
        # Verbatim completion
        inputs_c = tokenizer(context, return_tensors='pt').to(model.device)
        max_new  = len(target.split()) + 5
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
    results.to_csv(f'{paths.results}/results_{title}_verbatim.csv', index=False)

# Statistics
results['rougel'].hist()
results['perplexity'].hist()

#%%