import torch

from tqdm import tqdm

from metrics import evaluate

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
    (best_end2, best_start2), max_prob2 = find_best_answer(start_probs[split_index+1:], end_probs[split_index+1:])
    if max_prob>max_prob2:
        return  (best_end, best_start), max_prob
    else:
        return (best_end2, best_start2), max_prob2


def predict(model, valid_loader, args, tokenizer, progress=False, limit=None, decode_mode='greedy'):
    model.eval()
    all_outputs = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for i, (ori_sen, token, token_type, start, end, insert_pos, start_ner, end_ner) in enumerate(tqdm(valid_loader)):
            input_mask = (token > 0).to(device)
            token, input_mask, token_type = \
                token.to(device), input_mask.to(device), token_type.to(device)
            if start[0] is not None:
                start, end, insert_pos, start_ner, end_ner = start.to(device), end.to(device), insert_pos.to(device), start_ner.to(device), end_ner.to(device)
            outputs = model(input_ids=token, attention_mask=input_mask, token_type_ids=token_type,
                            start=None, end=end, insert_pos=insert_pos, start_ner=start_ner,
                            end_ner=end_ner)
            start_logits, end_logits, insert_pos_logits, start_ner_logits, end_ner_logits = \
                outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
            # 解码出真实label
            for i in range(len(token)):
                try:
                    split_index = len(ori_sen[i][1])+1
                    (best_start, best_end), max_prob = find_best_answer_for_passage(start_logits[i], end_logits[i], split_index)
                except:
                    print('something error!')
                    pass
                token_seq = ''.join(ori_sen[i])
                info_pos = (best_start.cpu().numpy()[0], best_end.cpu().numpy()[0])
                token_subseq = token_seq[info_pos[0]:(info_pos[1]+1)]
                text = token_subseq
                # print('关键信息检测 ', text)
                context_len = sum(token_type[i].cpu().numpy() == 0)
                input_len = len(token_seq)
                if info_pos[1] == 0 or len(text) == 0 or text in token_seq[context_len::]:
                    all_outputs.append(token_seq[context_len:input_len-1])
                    continue
                # 插入和指代做比较
                insert_prob = max(insert_pos_logits[i])
                # 指代概率
                (best_start, best_end), max_ner_prob = find_best_answer(start_ner_logits[i], end_ner_logits[i])
                if  best_start>=context_len and max_ner_prob>insert_prob:
                    pos = (best_start.cpu().numpy()[0], best_end.cpu().numpy()[0])
                    token_subseq = token_seq[pos[0]:(pos[1] + 1)]
                    replace_text = token_subseq
                    rewritten_text = token_seq[context_len:pos[0]]+text+token_seq[pos[1]+1:input_len-1]
                    all_outputs.append(rewritten_text)
                    continue

                if insert_pos_logits[i].argmax() >= context_len:
                    # 插入字符串
                    insert_pos = insert_pos_logits[i].argmax().cpu().numpy()
                    rewritten_text = token_seq[context_len:insert_pos]+text+\
                                         token_seq[insert_pos:input_len-1]
                    all_outputs.append(rewritten_text)
                    pass
                else:# best_start<context_len:
                    # 指代消歧
                    all_outputs.append(token_seq[context_len:input_len-1])

    # print(all_outputs)
    # if progress:
    #     tq.close()
    return all_outputs


def validate(model, valid_loader, valid_df, args, tokenizer, ner_index, save_result=False, progress=False, limit=None,
             decode_mode='greedy'):
    predictions = predict(model, valid_loader, args, tokenizer, progress=True, limit=limit, decode_mode=decode_mode)
    valid_label = valid_df['eval_label'].tolist()
    print_label = valid_df['label'].tolist()
    a = valid_df['a'].tolist()
    b = valid_df['b'].tolist()
    current = valid_df['current'].tolist()
    # print(len(predictions),len(valid_label))
    predictions = [' '.join(x) for x in predictions]
    valid_metric = evaluate(predictions, valid_label)
    print(valid_metric)
    print('------------')
    for i, (a, b, current, p, l) in enumerate(zip(a, b, current, predictions, print_label)):
        print(a,' | ', b,' | ', current,' | ', p.replace(' ',''),' | ', l)
        if i >= args.print_num:
            break
    return valid_metric