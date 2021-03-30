from rouge import Rouge 


def evaluate(pred, label):
    rouge = Rouge()
    rouge_score = rouge.get_scores(pred, label, avg=True)
    em_count = 0
    for p, l in zip(pred, label):
        if p==l:
            em_count+=1
    rouge_score['em']=em_count/len(pred)
    return rouge_score
