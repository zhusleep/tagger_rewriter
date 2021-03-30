
import numpy as np
import random
import torch
from datetime import datetime
import json


def set_seed(seed=6750):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model, path):
    state = torch.load(str(path))
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))
    return state, state['loss']


def save_model(model, ep, step, loss, model_path):
    torch.save({
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'epoch': ep,
        'step': step,
        'loss': loss
    }, str(model_path))

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    lr = lr[0]

    return lr


def write_event(log, step: int, epoch=None, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    data['epoch']=epoch
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def find_best_answer(start_probs, end_probs):
    best_start, best_end, max_prob = -1, -1, 0

    start_probs, end_probs = start_probs.unsqueeze(0), end_probs.unsqueeze(0)
    prob_start, best_start = torch.max(start_probs, 1)
    prob_end, best_end = torch.max(end_probs, 1)
    num = 0
    while True:
        if num > 3:
            break
        if best_end >= best_start:
            break
        else:
            start_probs[0][best_start], end_probs[0][best_end] = 0.0, 0.0
            prob_start, best_start = torch.max(start_probs, 1)
            prob_end, best_end = torch.max(end_probs, 1)
        num += 1
    max_prob = prob_start * prob_end

    if best_start <= best_end:
        return (best_start, best_end), max_prob
    else:
        return (best_end, best_start), max_prob


def find_best_answer_for_passage(start_probs, end_probs, split_index):
    (best_end, best_start), max_prob = find_best_answer(start_probs[0:split_index], end_probs[0:split_index])
    (best_end2, best_start2), max_prob2 = find_best_answer(start_probs[split_index+2:], end_probs[split_index+2:])
    if max_prob>max_prob2:
        return  (best_end, best_start), max_prob
    else:
        return (best_end2, best_start2), max_prob2
