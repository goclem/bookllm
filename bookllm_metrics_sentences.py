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

#%% LOADS TOKENIZER AND MODELS

# Parameters
llama_backbone = 'meta-llama/Meta-Llama-3-70B'
ner_backbone   = 'Jean-Baptiste/camembert-ner-with-dates'
device         = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

''' #! For testing only
llama_backbone = 'meta-llama/Meta-Llama-3-8B'
context = 'The quick brown fox jumps over the'
target  = 'lazy dog'
'''

# Llama token
with open(f'{paths.data}/hf_token.txt', 'r') as file:
    token = file.read().strip()

# Llama model and tokenizer
if 'llama_tokenizer' or 'llama_model' not in dir():
    llama_tokenizer = AutoTokenizer.from_pretrained(
        llama_backbone,
        token=token)
    llama_tokenizer.pad_token    = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = 'left'
    llama_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    llama_model  = AutoModelForCausalLM.from_pretrained(
        llama_backbone, 
        quantization_config=llama_config,
        device_map='auto',
        token=token)

# NER pipeline
if 'ner_pipeline' not in dir():
    ner_tokenizer = AutoTokenizer.from_pretrained(
        ner_backbone, 
        token=token)
    ner_model = AutoModelForTokenClassification.from_pretrained(
        ner_backbone,
        token=token)
    ner_pipeline  = pipeline('ner', model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy='simple')

#%% FORMATS DATA

# Books
books = search_data(f'{paths.data}', pattern='pdf$', kind='file') #! Change folder
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

    # Computes metrics
    results  = []
    rougel   = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    progress = tqdm.tqdm(loader, total=len(loader), desc='Verbatim test')
    running_rougel, running_perplex, running_n = 0, 0, 0
    
    for batch in progress:

        # Given batch size of 1
        context, target = batch[0]
        
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
        entities_context    = [(entity['word'], entity['entity_group']) for entity in ner_pipeline(context)]
        entities_target     = [(entity['word'], entity['entity_group']) for entity in ner_pipeline(target)]
        entities_prediction = [(entity['word'], entity['entity_group']) for entity in ner_pipeline(prediction)]
        
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
            'entities_context':entities_context,
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

#%%