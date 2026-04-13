#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@description: Utilities for the bookllm project
@author: Clement Gorin
@contact: clement.gorin@univ-paris1.fr
'''

#%% HEADER

# Packages
import argparse
import numpy as np
import os
import re
import shutil

#%% PATHS UTILITIES

# Laptop
if os.getlogin() == 'root':
    home  = os.path.expanduser('~')
    paths = argparse.Namespace(
        desktop=f'{home}/Desktop',
        downloads=f'{home}/Downloads',
        data=f'{home}/Desktop/temporary/bookllm',
        results=f'{home}/Library/CloudStorage/Dropbox/consultancy/bookllm/results'
    )
    del home

# Server
if os.getlogin() == 'clgorin':
    home  = os.path.expanduser('~')
    paths = argparse.Namespace(
        desktop=f'{home}/Desktop',
        downloads=f'{home}/Downloads',
        data=f'{home}/Desktop/temporary/bookllm',
        results=f'{home}/Library/CloudStorage/Dropbox/consultancy/bookllm/results'
    )
    del home

#%% FILE UTILITIES

def search_data(directory:str='../data', pattern:str='.*', kind:str='file') -> np.ndarray:
    '''Sorted list of files in a directory matching a regular expression'''
    output = list()
    for root, dirnames, filenames in os.walk(directory):
        if kind == 'file':
            for filename in filenames:
                output.append(os.path.join(root, filename))
        if kind == 'directory':
            for dirname in dirnames:
                output.append(os.path.join(root, dirname))
    output = list(filter(re.compile(pattern).search, output))
    output.sort()
    output = np.array(output)
    return output

def reset_folder(path:str, remove:bool=False) -> None:
    '''Resets a folder'''
    if remove:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(path)

def filename(filepath:str, extension=False) -> str:
    '''Extracts file name'''
    filename = os.path.basename(filepath)
    if extension is False:
        filename = os.path.splitext(filename)[0]
    return filename

#%%