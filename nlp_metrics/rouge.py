'''calculate ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score'''
import numpy as np
import nltk
from .utils import Coder

def calc_recall(sent_gt, sent_pred, n):
    '''calculate RECALL with ROUGE-N

    ROUGE-RECALL = number_of_overlapping_words / total_words_in_ground_truth_sentence

    Example:
        >>> sent_gt = 'the cat was under the bed .'
        >>> sent_pred = 'the cat was found under the bed .'
        >>> # calculate RECALL
        >>> calc_recall(sent_gt, sent_pred, n=1)
        >>> # 1.0
    
    Args:
        sent_gt: grand truth sentence string
        sent_pred: predicted sentence string
        n: int. n-gram

    Return:
        float. recall score with ROUGE-N
    '''
    ds = [sent_gt, sent_pred]
    coder = Coder()
    coder.build(ds)

    idx_gt = coder.encode(sent_gt)
    idx_pred = coder.encode(sent_pred)

    n_gram_gt = list(nltk.ngrams(idx_gt, n=n))
    n_gram_pred = list(nltk.ngrams(idx_pred, n=n))

    pairs = list(set(n_gram_gt + n_gram_pred))

    # count number of overlapping words
    cnt_overlapping = 0
    for w in pairs:
        if w in n_gram_gt and w in n_gram_pred:
            cnt_overlapping += 1
    
    # calculate recall
    recall = cnt_overlapping / len(n_gram_gt)

    return recall

def calc_precision(sent_gt, sent_pred, n):
    '''calculate PRECISION with ROUGE-N

    ROUGE-PRECISION = number_of_overlapping_words / total_words_in_predicted_sentence

    Exmaple:
        >>> sent_gt = 'the cat was under the bed .'
        >>> sent_pred = 'the cat was found under the bed .'
        >>> # calculate PRECISION
        >>> calc_precision(sent_gt, sent_pred, n=1)
        >>> # 0.86

    Args:
        sent_gt: grand truth sentence string
        sent_pred: predicted sentence string
        n: int. n-gram

    Return:
        float. precision score with ROUGE-N
    '''
    ds = [sent_gt, sent_pred]
    coder = Coder()
    coder.build(ds)

    idx_gt = coder.encode(sent_gt)
    idx_pred = coder.encode(sent_pred)

    n_gram_gt = list(nltk.ngrams(idx_gt, n=n))
    n_gram_pred = list(nltk.ngrams(idx_pred, n=n))

    pairs = list(set(n_gram_gt + n_gram_pred))

    # count number of overlapping words
    cnt_overlapping = 0
    for w in pairs:
        if w in n_gram_gt and w in n_gram_pred:
            cnt_overlapping += 1
    
    # calculate precision
    precision = cnt_overlapping / len(n_gram_pred)

    return precision
