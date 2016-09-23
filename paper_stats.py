
"""This script contains a number of functions that find
different statistics about a given text file. It includes
functions at the end that run each individual function and
aggregate the outputs. This can be used to generate feature
arrays describing each input text"""

import re
import pandas as pd
import numpy as np
import pickle
import csv
from textstat.textstat import *

def word_count(paper):
    paper = paper.split()
    return len(paper)

def char_count(word):
    return len(word)

def avg_word_length(paper):
    words = word_count(paper)
    p=paper.split()
    char_total = 0
    for word in p:
        char_total += char_count(word)
    return round(float(char_total)/float(words), 3)

def max_word_length(paper):
    p=paper.split()
    max_length = 0
    for word in p:
        if len(word) > max_length:
            max_length = len(word)
    return max_length

def avg_syllables_per_word(paper):
    return textstat.avg_syllables_per_word(paper)

def avg_letters_per_word(paper):
    return textstat.avg_letter_per_word(paper)

def num_difficult_words(paper):
    return textstat.difficult_words(paper)

def num_polysyllable(paper):
    return textstat.polysyllabcount(paper)

def num_sentences(paper):
    return textstat.sentence_count(paper)

def ref_count_super_basic(paper):
    return paper.count(')')

def url_count(paper):
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    parsed_text = re.sub(giant_url_regex, 'URL', paper)
    return paper.count('URL')

def basic_stats(paper):
    """Calculates a number of basic statistics about each paper and
    returns them as a list"""
    return [word_count(paper), avg_word_length(paper), max_word_length(paper), avg_syllables_per_word(paper) /
            avg_letters_per_word(paper), num_difficult_words(paper), num_polysyllable(paper), num_sentences(paper)] #/

          # ref_count_super_basic(paper), url_count(paper)]

def readability(paper):
    """Calculates a number of different readability metrics
    for each paper and returns them as a list"""
    fkg = textstat.flesch_kincaid_grade(paper)
    fre = textstat.flesch_reading_ease(paper)
    dcr = textstat.dale_chall_readability_score(paper)
    smg = textstat.smog_index(paper)
    cli = textstat.coleman_liau_index(paper)
    ari = textstat.automated_readability_index(paper)
    return [fkg,fre,dcr,smg,cli,ari]

def avg_readability(readability_scores):
    """Returns the mean readability score & its SD across
    all scores for a given essay."""
    readability = np.array(readability_scores)
    mean = np.mean(readability)
    sd = np.std(readability)
    return [mean, sd]

def get_stats(papers):
    paper_stats = []
    for p in papers:
        rd = readability(p)
        avgs = avg_readability(rd)
        bs = basic_stats(p)
        stats = rd+avgs+bs
        for i in range(0, len(stats)):
            stats[i] = round(stats[i], 3)
        paper_stats.append(stats)
    return pd.DataFrame(np.array(paper_stats))
