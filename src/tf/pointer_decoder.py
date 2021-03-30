from utils import find_best_answer, find_best_answer_for_passage


def pointer_decode(data, model):
    all_outputs = []
    for model_inputs, ori_sen in data:
        token, token_type, start_labels, end_labels, insert_pos_labels, start_ner_labels, end_ner_labels = model_inputs
        start_logits, end_logits, insert_pos_logits, start_ner_logits, end_ner_logits = model(model_inputs)

        for i in range(len(token)):
            try:
                split_index = len(ori_sen[i][1])+1
                #print(split_index)
                (best_start, best_end), max_prob = find_best_answer_for_passage(start_logits[i], end_logits[i], split_index)
            except:
                print('something error!')
                pass
            token_seq = ''.join(ori_sen[i])
            info_pos = (best_start, best_end)
            #print(info_pos)
            token_subseq = token_seq[info_pos[0]:(info_pos[1]+1)]
            text = token_subseq
            #print('关键信息检测 ', text)
            context_len = sum(token_type[i] == 0)
            input_len = len(token_seq)
            if info_pos[1] == 0 or len(text) == 0 or text in token_seq[context_len::]:
                all_outputs.append(token_seq[context_len:input_len-1])
                continue
            # 插入和指代做比较
            insert_prob = max(insert_pos_logits[i].numpy())
            #print(f"insert prob: {insert_prob}")
            #print(insert_pos_logits[i].numpy().argmax())
            # 指代概率
            (best_start, best_end), max_ner_prob = find_best_answer(start_ner_logits[i], end_ner_logits[i])
            #print(f"max ner prob: {max_ner_prob}")
            if  best_start>=context_len and max_ner_prob>insert_prob or insert_pos_logits[i].numpy().argmax() == 0:
                pos = (best_start, best_end)
                token_subseq = token_seq[pos[0]:(pos[1] + 1)]
                replace_text = token_subseq
                rewritten_text = token_seq[context_len:pos[0]]+text+token_seq[pos[1]+1:input_len-1]
                all_outputs.append(rewritten_text)
                continue

            if insert_pos_logits[i].numpy().argmax() >= context_len:
                # 插入字符串
                insert_pos = insert_pos_logits[i].numpy().argmax()
                rewritten_text = token_seq[context_len:insert_pos]+text+\
                                        token_seq[insert_pos:input_len-1]
                all_outputs.append(rewritten_text)
                pass
            else:# best_start<context_len:
                # 指代消歧
                all_outputs.append(token_seq[context_len:input_len-1])

    #print(all_outputs)
    return all_outputs


def validate(model, valid_generator, valid_df, metrics_fn, example_limit=5):
    predictions = pointer_decode(valid_generator, model)

    valid_label = valid_df['eval_label'].tolist()
    print_label = valid_df['label'].tolist()
    a = valid_df['a'].tolist()
    b = valid_df['b'].tolist()
    current = valid_df['current'].tolist()
    # print("##",len(predictions),len(valid_label))
    predictions = [' '.join(x) for x in predictions]
    valid_metric = metrics_fn(predictions, valid_label)
    print(valid_metric)
    print('------------')
    for i, (a, b, current, p, l) in enumerate(zip(a, b, current, predictions, print_label)):
        print(a,' | ', b,' | ', current,' | ', p.replace(' ',''),' | ', l)
        if i >= example_limit:
            break
    return valid_metric