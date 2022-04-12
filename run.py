from transformers import BertTokenizer,BertConfig
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from transformers.optimization import get_cosine_schedule_with_warmup,AdamW
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
import logging

from model import TaggerRewriteModel
from utils import seed_everything,set_logger
from config import set_train_args
from data_utils import TaggerRewriterDataset,tagger_collate_fn
from decode import validate


args = set_train_args()
seed_everything(args.seed)
#记录tensorboard和日志
writer = SummaryWriter(os.path.join(args.tensorboard_dir, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
set_logger(args.log_dir)
device='cuda' if torch.cuda.is_available() else 'cpu'
#读取和划分数据集
df = pd.read_table('./data/dialog-rewrite/corpus.txt', sep="\t\t", names=['a', 'b', 'current', 'label'], dtype=str,
                   encoding='utf-8')
df.dropna(how='any', inplace=True)
train_length = int(len(df) * 0.9)
train_df = df.iloc[:train_length].iloc[:, :]
valid_df = df.iloc[train_length:]
valid_df['eval_label'] = valid_df['label'].apply(lambda x: ' '.join(list(x)))
#训练集处理
tokenizer = BertTokenizer.from_pretrained("voidful/albert_chinese_tiny")
train_set = TaggerRewriterDataset(train_df, tokenizer)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,collate_fn=tagger_collate_fn)
valid_set = TaggerRewriterDataset(valid_df, tokenizer, valid=True)
valid_index = np.array(valid_set.valid_index)
valid_df = valid_df.reset_index().loc[valid_index, :]
ner_index = np.array(valid_set.label_type) == 1
valid_loader = DataLoader(valid_set, batch_size=args.batch_size,shuffle=False, collate_fn=tagger_collate_fn)
#模型设置
config = BertConfig("voidful/albert_chinese_tiny")
config.num_labels = 5
model = TaggerRewriteModel(config, None)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {"params": [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=args.lr, eps=args.adam_epsilon)
t_total = int(len(train_loader))* args.epoch
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0.1 * t_total,
                                            num_training_steps=t_total)
#模型训练
criterion = CrossEntropyLoss().cuda()
best_valid_score=0.0
losses = 0.0
logging.info("--------Start Training!--------")
for epoch in range(1, args.epoch):
    model.train()
    for i, (ori_sen, token, token_type, start, end, insert_pos, start_ner, end_ner) in enumerate(tqdm(train_loader)):
        input_mask = (token > 0).to(device)
        token, input_mask, token_type, start, end, insert_pos, start_ner, end_ner = \
            token.to(device), input_mask.to(device), token_type.to(device), start.to(
                device), end.to(device), insert_pos.to(device), start_ner.to(device), end_ner.to(device)
        outputs = model(input_ids=token, attention_mask=input_mask, token_type_ids=token_type,
                        start=start, end=end, insert_pos=insert_pos, start_ner=start_ner,
                        end_ner=end_ner)
        loss = outputs[0]
        losses+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    train_loss=losses/len(train_loader)
    logging.info('Epoch: {},train loss: {}'.format(epoch, train_loss))
    writer.add_scalar('Training/training loss', train_loss, epoch)## tensorboard --logdir "./runs"启动

    valid_metrics = validate(model, valid_loader, valid_df, args, tokenizer, ner_index)
    logging.info("Epoch: {}, rouge1 f1: {},rouge2 f1: {},rouge-L f1: {}".format(epoch, valid_metrics['rouge-1']['f'],
                                                                                                   valid_metrics['rouge-2']['f'],
                                                                                                   valid_metrics['rouge-l']['f']
                                                                                                   ))
    writer.add_scalar('Validation/rouge-1 f1', valid_metrics['rouge-1']['f'], epoch)
    writer.add_scalar('Validation/rouge-2 f1', valid_metrics['rouge-2']['f'], epoch)
    writer.add_scalar('Validation/rouge-l f1', valid_metrics['rouge-l']['f'], epoch)
    current_score = valid_metrics['rouge-1']['f']
    if current_score > best_valid_score:
        print("Epoch: {}".format(epoch)+'保存模型')
        torch.save(model.state_dict(),'./best_model.pkl')

