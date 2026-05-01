#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Computes metrics for the BookLLM project
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
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from bookllm_utilities import *

# Parameters
llama_backbone = 'meta-llama/Meta-Llama-3-70B'
ner_backbone   = 'Jean-Baptiste/camembert-ner-with-dates'
# backbone = 'meta-llama/Meta-Llama-3-8B'  #! For testing
device   = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

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

    def __init__(self, sentences:list, n_context:int=3, n_target:int=2, stride:int=2, max_samples:int=None, seed:int=0) -> None:
        self.sentences  = sentences
        self.n_context  = n_context
        self.n_target   = n_target
        self.stride     = stride
        self.batch_size = 1 # Hard coded given memory constraints
        self.indices    = range(n_context, len(sentences) - n_target + 1, self.stride)
        if max_samples is not None and len(self.indices) > max_samples:
            rng = np.random.default_rng(seed)
            self.indices = rng.choice(list(self.indices), size=max_samples, replace=False)

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

#%% LOADS TOKENIZER AND MODELS

with open(f'{paths.data}/llama_token.txt', 'r') as file:
    token = file.read().strip()

# Llama model
if 'llama_model' not in dir():
    llama_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    llama_model  = AutoModelForCausalLM.from_pretrained(
        llama_backbone, 
        quantization_config=llama_config,
        device_map='auto',
        token=token)

# Llama tokeniser
if 'llama_tokenizer' not in dir():
    llama_tokenizer = AutoTokenizer.from_pretrained(
        llama_backbone,
        token=token)
    llama_tokenizer.pad_token    = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = 'left'

# NER pipeline
if 'ner_pipeline' not in dir():
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_backbone)
    ner_model     = AutoModelForTokenClassification.from_pretrained(ner_backbone)
    ner_pipeline  = pipeline('ner', model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy='simple')

#%% FORMATS DATA

# Books
books = search_data(f'{paths.data}', pattern='pdf$', kind='file')
label = 'sentence_results' # Label for the output file

# Iterates over books
for book in books:

    # Skips of results already exist
    dstfile = f'{paths.results}/results_{filename(book)}_{label}.csv'
    if os.path.exists(dstfile):
        print(f'{dstfile} already exists, skipping.')
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
    loader = SentenceLoader(
        sentences, 
        n_context=3, 
        n_target=2, 
        stride=2, 
        max_samples=None,
        seed=0
    )

    ''' #! Testing
    context = 'The quick brown fox jumps over the'
    target  = 'lazy dog'
    '''

    # Computes metrics
    results  = []
    rougel   = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    progress = tqdm.tqdm(loader, total=len(loader), desc='Verbatim test')
    running_rougel, running_perplex, running_n = 0, 0, 0
    
    for batch in progress:
        context, target = batch[0] # Given batch size of 1
        
        # Verbatim completion
        context_tokens = llama_tokenizer(context, return_tensors='pt').to(llama_model.device)
        target_tokens  = llama_tokenizer(target).input_ids
        with torch.no_grad():
            outputs = llama_model.generate(
                **context_tokens, 
                max_new_tokens=len(target_tokens) + 5, 
                do_sample=False, 
                pad_token_id=llama_tokenizer.eos_token_id)
        prediction   = outputs[0, context_tokens.input_ids.shape[-1]:] # Remove context tokens
        prediction   = llama_tokenizer.decode(prediction, skip_special_tokens=True)
        rougel_score = rougel.score(target, prediction)['rougeL'].fmeasure
        
        # Perplexity score
        full_tokens = llama_tokenizer(context + ' ' + target, return_tensors='pt').to(llama_model.device)
        mask_tokens = full_tokens.input_ids.clone()
        mask_tokens[:, :context_tokens.input_ids.shape[-1]] = -100 # Mask context tokens
        with torch.no_grad():
            perplex_score = torch.exp(llama_model(**full_tokens, labels=mask_tokens).loss).item()
        
        # Named entity recognition
        entities_target     = ner_pipeline(target)
        entities_target     = [(entity['word'], entity['entity_group']) for entity in entities_target]
        entities_prediction = ner_pipeline(prediction)
        entities_prediction = [(entity['word'], entity['entity_group']) for entity in entities_prediction]

        # Collects results
        running_rougel  += rougel_score
        running_perplex += perplex_score
        running_n       += 1
        results.append({
            'context':context, 
            'target':target, 
            'prediction':prediction, 
            'rougel':rougel_score, 
            'perplexity':perplex_score,
            'target_tokens':len(target_tokens),
            'entities_target':entities_target,
            'entities_prediction':entities_prediction})
        
        # Prints average statistics
        progress.set_postfix({
            'rougel':     f'{running_rougel/running_n:.3f}', 
            'perplexity': f'{running_perplex/running_n:.3f}'
        })

    # Saves results
    results = pd.DataFrame(results)
    results.to_csv(dstfile, index=False)

#%% EXPLORES RESULTS

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
