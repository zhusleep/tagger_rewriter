
import numpy as np
import random
from datetime import datetime
import json
from rouge import Rouge 


def find_best_answer(start_probs, end_probs):
    best_start, best_end, max_prob = -1, -1, 0

    start_probs, end_probs = start_probs.numpy(), end_probs.numpy()
    prob_start, best_start = start_probs.max(), start_probs.argmax()
    prob_end, best_end = end_probs.max(), end_probs.argmax()
    num = 0
    while True:
        if num > 3:
            break
        if best_end >= best_start:
            break
        else:
            start_probs[best_start], end_probs[best_end] = 0.0, 0.0
            prob_start, best_start = start_probs.max(), start_probs.argmax()
            prob_end, best_end = end_probs.max(), end_probs.argmax()
        num += 1
    max_prob = prob_start * prob_end

    if best_start <= best_end:
        return (best_start, best_end), max_prob
    else:
        return (best_end, best_start), max_prob


def find_best_answer_for_passage(start_probs, end_probs, split_index):
    (best_end, best_start), max_prob = find_best_answer(start_probs[0:split_index], end_probs[0:split_index])
    (best_end2, best_start2), max_prob2 = find_best_answer(start_probs[split_index+1:], end_probs[split_index+1:])
    if max_prob>max_prob2:
        return  (best_end, best_start), max_prob
    else:
        return (best_end2+split_index+1, best_start2+split_index+1), max_prob2


def metrics_fn(pred, label):
    rouge = Rouge()
    rouge_score = rouge.get_scores(pred, label, avg=True)
    em_count = 0
    for p, l in zip(pred, label):
        if p==l:
            em_count+=1
    rouge_score['em']=em_count/len(pred)
    return rouge_score
