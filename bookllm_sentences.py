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

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

class SentenceLoader:

    def __init__(self, sentences:list, n_context:int=3, n_target:int=1, stride:int=1, sample_size:int=None, batch_size:int=1, seed:int=0) -> None:
        self.sentences  = sentences
        self.n_context  = n_context
        self.n_target   = n_target
        self.stride     = stride
        self.batch_size = batch_size
        self.indices    = range(n_context, len(sentences) - n_target + 1, self.stride)
        if sample_size is not None:
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(list(self.indices), size=sample_size, replace=False)

    def __len__(self):
        return int(math.ceil(len(self.indices) / self.batch_size))

    def __iter__(self):
        batch = []
        for i in self.indices:
            context = ' '.join(self.sentences[max(0, i-self.n_context):i])
            target  = ' '.join(self.sentences[i:i+self.n_target])
            batch.append((context, target))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def pval_to_stars(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ''

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
    dstfile = f'{paths.results}/results_{filename(book)}_results.csv'
    if os.path.exists(dstfile):
        print(f'{book} already exists, skipping.')
        continue

    # Loads text
    pages = fitz.open(book)
    pages = [page for page in pages if page.get_text().strip()]

    # Extracts sentences
    sentences = []
    for page in pages:
        text = page.get_text()
        text = clean_text(text)
        text = split_sentences(text)
        sentences.extend(text)
    del page, text

    # Initialises loader
    loader = SentenceLoader(sentences, n_context=3, n_target=2, stride=2, batch_size=1, sample_size=None)
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

#%% RESULTS

# Packages
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from bookllm_utilities import *
from matplotlib.backends.backend_pdf import PdfPages

colours = {'target': 'blue', 'control': 'orange'}

with PdfPages(f'{paths.desktop}/perplexity_distributions.pdf') as pdf:
    for author in ['carrere', 'hawkins', 'lenoir', 'plenel']:
        # Loads data
        target  = pd.read_csv(f'{paths.results}/results_{author}_target.csv')
        control = pd.read_csv(f'{paths.results}/results_{author}_control.csv')

        # Filters out short sentences (less than 10 tokens)
        keep_t  = [len(tokens) > 10 for tokens in tokenizer(target['target'].tolist()).input_ids]
        keep_c  = [len(tokens) > 10 for tokens in tokenizer(control['target'].tolist()).input_ids]
        target  = target[['perplexity']][keep_t]
        control = control[['perplexity']][keep_c]

        # Smooths perplexity distributions
        low  = pd.concat([target['perplexity'], control['perplexity']]).min()
        high = pd.concat([target['perplexity'], control['perplexity']]).max()
        x    = np.linspace(low, high, 1000)
        kde_c = stats.gaussian_kde(control['perplexity'], bw_method=0.05)
        kde_t = stats.gaussian_kde(target['perplexity'],  bw_method=0.05)

        # Test
        median_c = control['perplexity'].median()
        median_t = target['perplexity'].median()
        stat_p, pval_p = stats.ks_2samp(target['perplexity'], control['perplexity'])

        # Plots distribution
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(control['perplexity'], bins=250, alpha=0.25, density=True, color=colours['control'], label=f'Control ({len(control)} obs.)')
        ax.hist(target['perplexity'],  bins=250, alpha=0.25, density=True, color=colours['target'],  label=f'Target ({len(target)} obs.)')
        ax.plot(x, kde_c(x), linewidth=1, color=colours['control'])
        ax.plot(x, kde_t(x), linewidth=1, color=colours['target'])
        ax.axvline(median_c, color=colours['control'], linestyle='dashed', linewidth=1)
        ax.axvline(median_t, color=colours['target'],  linestyle='dashed', linewidth=1)
        ax.text(median_c + 0.1, ax.get_ylim()[1] * 0.95, f'{median_c:.2f}', color=colours['control'], ha='left', va='top', fontsize=9, rotation=90)
        ax.text(median_t + 0.1, ax.get_ylim()[1] * 0.95, f'{median_t:.2f}', color=colours['target'],  ha='left', va='top', fontsize=9, rotation=90)
        ax.set_xlim(low, 25)
        ax.set_xlabel('Perplexity')
        ax.set_ylabel('Density')
        ax.set_title(f'KS test: {stat_p:.3f}{pval_to_stars(pval_p)}')
        ax.legend()
        plt.suptitle(f'{author.capitalize()}', fontsize=20)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.show()
        plt.close(fig)

#%% RESULTS

# Packages
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from bookllm_utilities import *
from matplotlib.backends.backend_pdf import PdfPages

colours = {'target': 'blue', 'control': 'orange'}

for author in ['carrere', 'hawkins', 'lenoir', 'plenel']:
    target  = pd.read_csv(f'{paths.results}/results_{author}_target.csv')
    control = pd.read_csv(f'{paths.results}/results_{author}_control.csv')
    print(f"{author} target:  {target['rougel'].sort_values(ascending=False)[:100].median():.2f}")
    print(f"{author} control: {control['rougel'].sort_values(ascending=False)[:100].median():.2f}")
    #print(f"{author} target:  {target['perplexity'].sort_values(ascending=True)[:100].median():.2f}")
    #print(f"{author} control: {control['perplexity'].sort_values(ascending=True)[:100].median():.2f}")
