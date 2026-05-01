
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Explores results for the BookLLM project
@author: Clement Gorin
@contact: clement.gorin@univ-paris1.fr
'''

# Packages
import fitz
import math
import pandas as pd
import re
import torch
import tqdm

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
