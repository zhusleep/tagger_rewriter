from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import torch


class TaggerRewriterDataset(Dataset):

    def __init__(self, df, tokenizer, valid=False):
        self.a = df['a'].values.tolist()
        self.b = df['b'].values.tolist()
        self.is_valid = valid
        self.current = df['current'].values.tolist()
        self.label = df['label'].values.tolist()
        self._tokenizer = tokenizer
        self.ori_sentence = []
        self.sentence = []
        self.token_type = []
        self.pointer = []
        self.context_len = []
        self.valid_index = []
        self.valid_label = []
        self.label_type = []
        self.generate_label()

    def tokenize_chinese(self,sen):
        temp = []
        for word in sen:
            if word in self._tokenizer.vocab:
                temp.append(word)
            else:
                temp.append("[UNK]")
        return temp

    def generate_label(self):
        # 全部采用指针抽取
        # 根据改写的数据对原始数据进行标注
        # 去哪里    长城北路公园    在什么地方     长城北路公园在什么地方
        # 确实江西炒粉要用瓦罐汤 特产 没错是我老家的特产 没错江西炒粉是我老家的特产
        # 为什么讨厌张艺兴       我喜欢张艺兴 很可爱啊       我也喜欢他     我也喜欢张艺兴

        # start,end,insert,start_ner,end_ner
        drop_item = 0
        for i in range(len(self.a)):
            # 生成随机数决定样本要不要改写，否则把label作为current
            n = random.random()
            start, end, insert_pos, start_ner, end_ner = 0,0,0,0,0
            new_token_type = []
            context_new_input = ["[CLS]"]+self.tokenize_chinese(self.a[i])+["[SEP]"]+self.tokenize_chinese(self.b[i])+["[SEP]"]
            new_token_type.extend([0]*len(context_new_input))
            if n >= 0.3:
                utterance_token = self.tokenize_chinese(self.current[i])+["SEP"]
            else:
                utterance_token = self.tokenize_chinese(self.label[i])+["SEP"]
            new_input = context_new_input + utterance_token
            new_token_type.extend([1]*len(utterance_token))

            # 改写或者作为验证集时不对关键信息进行抽取
            if self.is_valid or n<0.3 and False:
                # 改写的负样本
                if self.is_valid:
                    _label = [None] * 5
                else:
                    _label = [0]*5
                self.pointer.append(_label)
                self.sentence.append(self._tokenizer.convert_tokens_to_ids(new_input))
                self.token_type.append(new_token_type)
                self.context_len.append(context_new_input)
                self.ori_sentence.append([',', self.a[i], ',' + self.b[i], ',', self.current[i], ','])
                self.valid_index.append(i)
                self.valid_label.append(self.label[i])
                self.label_type.append([0, 0])
                continue
            # 获取四个指针信息
            insert = True
            # 如果原始语句所有词汇都在改写中，则改写为插入新语句
            for word in self.current[i]:
                if word not in self.label[i]:
                    insert = False
            # -----寻找增加的信息------------------
            text_start, text_end = 0, 0
            for j in range(len(self.label[i])):
                if j >= len(self.current[i]):
                    text_start = j
                    break
                if self.current[i][j] == self.label[i][j]:
                    continue
                else:
                    text_start = j
                    break
            for j in range(len(self.label[i])):
                if j >= len(self.current[i]):
                    text_end = j
                    break
                if self.current[i][::-1][j] == self.label[i][::-1][j]:
                    continue
                else:
                    text_end = j
                    break
            text = self.label[i][text_start:(len(self.label[i]) - text_end)]
            # 获取插入文本及位置
            if text in self.a[i]:
                start = self.a[i].index(text) + 1
                end = start + len(text) - 1
            elif text in self.b[i]:
                start = self.b[i].index(text) + len(self.a[i]) + 2
                end = start + len(text) - 1
            else:
                drop_item += 1
                continue
            if insert:
                self.label_type.append(0)
                # 去哪里    长城北路公园    在什么地方     长城北路公园在什么地方
                insert_pos = len(self.current[i])-text_end + len(context_new_input)
            else:
                # 指代
                # 为什么讨厌张艺兴       我喜欢张艺兴 很可爱啊       我也喜欢他     我也喜欢张艺兴
                coref_start, coref_end = 0, 0
                for j in range(len(self.current[i])):
                    if self.current[i][j] == self.label[i][j]:
                        continue
                    else:
                        coref_start = j
                        break
                for j in range(len(self.current[i])):
                    if self.current[i][::-1][j] == self.label[i][::-1][j]:
                        continue
                    else:
                        coref_end = j
                        break
                self.label_type.append(1)
                start_ner = coref_start+len(context_new_input)
                end_ner = len(self.current[i])-coref_end+len(context_new_input)-1
            # print(self.a[i],self.b[i],self.current[i],self.label[i], text)
            # print(start,end,insert_pos,start_ner,end_ner)
            if self.is_valid:
                self.pointer.append(_label)
            else:
                self.pointer.append([start,end,insert_pos,start_ner,end_ner])
            self.sentence.append(self._tokenizer.convert_tokens_to_ids(new_input))
            self.token_type.append(new_token_type)
            self.context_len.append(context_new_input)
            self.ori_sentence.append(','+self.a[i]+','+self.b[i]+','+self.current[i]+',')
            self.valid_label.append(self.label[i])
            self.valid_index.append(i)
        print('数据总数 ', len(self.sentence), '丢弃样本数目 ', drop_item)
        print('信息插入', self.label_type.count(0))
        print('指代消歧义', self.label_type.count(1))


    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        return  self.ori_sentence[idx],\
                torch.LongTensor(self.sentence[idx]),  \
                torch.LongTensor(self.token_type[idx]),\
                self.pointer[idx][0],\
                self.pointer[idx][1],\
                self.pointer[idx][2],\
                self.pointer[idx][3],\
                self.pointer[idx][4]


def tagger_collate_fn(batch):
    # start, end, insert_pos, start_ner, end_ner = 0,0,0,0,0
    ori_sen, token, token_type, start, end,insert_pos,start_ner,end_ner = zip(*batch)
    token = pad_sequence(token, batch_first=True)
    token_type = pad_sequence(token_type, batch_first=True, padding_value=1)
    if start[0] is not None:
        start = torch.tensor(start)
        end = torch.tensor(end)
        insert_pos = torch.tensor(insert_pos)
        start_ner = torch.tensor(start_ner)
        end_ner = torch.tensor(end_ner)
    return ori_sen, token, token_type, start, end, insert_pos, start_ner, end_ner