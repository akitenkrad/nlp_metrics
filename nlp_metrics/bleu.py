'''calculate BLEU score'''
import numpy as np
import nltk
from .utils import Coder

def calc_bleu(sent_gt, sent_pred, n):
    '''calculate bleu score given grand truth and predicted sentences
    
    Args:
        sent_gt: grand truth sentence string
        sent_pred: predicted sentence string
        n: int. consider up to n-gram
        
    Return:
        float. BLEU score
    '''
    ds = [sent_gt, sent_pred]
    coder = Coder()
    coder.build(ds)
 
    # calculate i-gram precision
    #-----------------------------------------------------------------
    precision = 1.0
    for i in range(1, n+1):
        idx_gt = coder.encode(sent_gt)
        idx_pred = coder.encode(sent_pred)
        
        i_gram_gt = list(nltk.ngrams(idx_gt, n=i))
        i_gram_pred = list(nltk.ngrams(idx_pred, n=i))
        
        freq_gt = nltk.FreqDist(i_gram_gt)
        freq_pred = nltk.FreqDist(i_gram_pred)
        
        numerator = 0
        for word, gt_count in freq_gt.items():
            cnt_pred = freq_pred[word]
            cnt_gt = freq_gt[word]
            numerator += min(cnt_pred, cnt_gt)
        
        denominator = len(i_gram_pred)
         
        precision *= (numerator / denominator)
    precision = precision**(1/n)
    
    # calculate brevity penalty
    #-----------------------------------------------------------------
    penalty = min(1, np.exp(1 - (len(idx_gt)/len(idx_pred))))
    
    return penalty * precision
