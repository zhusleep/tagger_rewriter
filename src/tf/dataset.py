import random
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
import numpy as np

__all__ = ["generate_label", "data_generator"]


def tokenize_chinese(sen, tokenizer):
    temp = []
    for word in sen:
        if word.lower() in tokenizer._token_dict:
            temp.append(word.lower())
        else:
            temp.append("[UNK]")
    return temp


def generate_label(df, tokenizer, is_valid=False):
    D = []
    # 全部采用指针抽取
    # 根据改写的数据对原始数据进行标注
    # 去哪里    长城北路公园    在什么地方     长城北路公园在什么地方
    # 确实江西炒粉要用瓦罐汤 特产 没错是我老家的特产 没错江西炒粉是我老家的特产
    # 为什么讨厌张艺兴       我喜欢张艺兴 很可爱啊       我也喜欢他     我也喜欢张艺兴
    pass
    # start,end,insert,start_ner,end_ner
    a = df['a'].values.tolist()
    b = df['b'].values.tolist()
    is_valid = is_valid
    current = df['current'].values.tolist()
    label = df['label'].values.tolist()
    _tokenizer = tokenizer
    ori_sentence = []
    sentence = []
    token_type = []
    pointer = []
    context_len = []
    valid_index = []
    valid_label = []
    label_type = []


    drop_item = 0
    for i in range(len(a)):
        # 生成随机数决定样本要不要改写，否则把label作为current
        n = random.random()
        start, end, insert_pos, start_ner, end_ner = 0,0,0,0,0
        new_token_type = []
        context_new_input = ["[CLS]"]+tokenize_chinese(a[i], _tokenizer)+["[SEP]"]+tokenize_chinese(b[i], _tokenizer)+["[SEP]"]
        new_token_type.extend([0]*len(context_new_input))
        if n >= 0.3:
            utterance_token = tokenize_chinese(current[i], _tokenizer)+["SEP"]
        else:
            utterance_token = tokenize_chinese(label[i], _tokenizer)+["SEP"]
        new_input = context_new_input + utterance_token
        new_token_type.extend([1]*len(utterance_token))

        # 改写或者作为验证集时不对关键信息进行抽取
        if is_valid or n<0.3 and False:
            # 改写的负样本
            if is_valid:
                _label = [0] * 5
            else:
                _label = [0]*5
            pointer.append(_label)
            sentence.append(_tokenizer.tokens_to_ids(new_input))
            token_type.append(new_token_type)
            context_len.append(context_new_input)
            ori_sentence.append([',', a[i], ',' + b[i], ',', current[i], ','])
            valid_index.append(i)
            valid_label.append(label[i])
            label_type.append([0, 0])
            continue
        # 获取四个指针信息
        insert = True
        # 如果原始语句所有词汇都在改写中，则改写为插入新语句
        for word in current[i]:
            if word not in label[i]:
                insert = False
        # -----寻找增加的信息------------------
        text_start, text_end = 0, 0
        for j in range(len(label[i])):
            if j >= len(current[i]):
                text_start = j
                break
            if current[i][j] == label[i][j]:
                continue
            else:
                text_start = j
                break
        for j in range(len(label[i])):
            if j >= len(current[i]):
                text_end = j
                break
            if current[i][::-1][j] == label[i][::-1][j]:
                continue
            else:
                text_end = j
                break
        text = label[i][text_start:(len(label[i]) - text_end)]
        # 获取插入文本及位置
        if text in a[i]:
            start = a[i].index(text) + 1
            end = start + len(text) - 1
        elif text in b[i]:
            start = b[i].index(text) + len(a[i]) + 2
            end = start + len(text) - 1
        else:
            drop_item += 1
            continue
        if insert:
            label_type.append(0)
            # 去哪里    长城北路公园    在什么地方     长城北路公园在什么地方
            insert_pos = len(current[i])-text_end + len(context_new_input)
        else:
            # 指代
            # 为什么讨厌张艺兴       我喜欢张艺兴 很可爱啊       我也喜欢他     我也喜欢张艺兴
            coref_start, coref_end = 0, 0
            for j in range(len(current[i])):
                if current[i][j] == label[i][j]:
                    continue
                else:
                    coref_start = j
                    break
            for j in range(len(current[i])):
                if current[i][::-1][j] == label[i][::-1][j]:
                    continue
                else:
                    coref_end = j
                    break
            label_type.append(1)
            start_ner = coref_start+len(context_new_input)
            end_ner = len(current[i])-coref_end+len(context_new_input)-1
        # print(a[i],b[i],current[i],label[i], text)
        # print(new_input)
        # print(start,end,insert_pos,start_ner,end_ner)
        if is_valid:
            pointer.append(_label)
        else:
            pointer.append([start,end,insert_pos,start_ner,end_ner])
        sentence.append(_tokenizer.tokens_to_ids(new_input))
        token_type.append(new_token_type)
        context_len.append(context_new_input)
        ori_sentence.append([',', a[i], ',' + b[i], ',', current[i], ','])
        valid_label.append(label[i])
        valid_index.append(i)
    print('数据总数 ', len(sentence), '丢弃样本数目 ', drop_item)
    print('信息插入', label_type.count(0))
    print('指代消歧义', label_type.count(1))

    for ori_s, s, t, p, l in zip(ori_sentence, sentence, token_type, pointer, valid_label):
        D.append({"ori_sentence": ori_s, "sentence": s, "token_type": t, "pointer": p, "label": l})

    return D


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_ori_sentence = []
        batch_token_ids, batch_segment_ids = [], []
        batch_start, batch_end, batch_insert_pos, batch_start_ner, batch_end_ner = [], [], [], [], []
        for is_end, d in self.sample(random):
            ori_sentence, sentence, token_type, pointer = d["ori_sentence"], d["sentence"], d["token_type"], d["pointer"]
            token_ids, segment_ids = sentence, token_type
            batch_ori_sentence.append(ori_sentence)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            start, end, insert_pos, start_ner, end_ner = pointer
            batch_start.append(start)
            batch_end.append(end)
            batch_insert_pos.append(insert_pos)
            batch_start_ner.append(start_ner)
            batch_end_ner.append(end_ner)
            
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_ori_sentence = np.array(batch_ori_sentence)
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids, padding=1)
                batch_start = np.array(batch_start)
                batch_end = np.array(batch_end)
                batch_insert_pos = np.array(batch_insert_pos)
                batch_start_ner = np.array(batch_start_ner)
                batch_end_ner = np.array(batch_end_ner)
                yield [
                    batch_token_ids, batch_segment_ids,
                    batch_start, batch_end, batch_insert_pos, batch_start_ner, batch_end_ner], batch_ori_sentence
                batch_ori_sentence = []
                batch_token_ids, batch_segment_ids = [], []
                batch_start, batch_end, batch_insert_pos, batch_start_ner, batch_end_ner = [], [], [], [], []


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_table('../../data/dialog-rewrite/corpus.txt', sep="\t\t", names=['a','b','current','label'], dtype=str, engine='python')
    df.dropna(how='any', inplace=True)
    #df = df[:1000]
    dict_path = '../../albert_small_zh_google/vocab.txt'

    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    train_data = generate_label(df, tokenizer)
    
    train_generator = data_generator(train_data, batch_size=32)
