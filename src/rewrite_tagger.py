import argparse
import tqdm
import shutil
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from apex import amp
from pathlib import Path
import pandas as pd
import numpy as np
from src.model import TaggerRewriteModel
from src.dataset import TaggerRewriterDataset, tagger_collate_fn
from transformers import BertConfig, BertTokenizer, get_constant_schedule_with_warmup
from src.utils import load_model, get_learning_rate, find_best_answer, save_model, write_event, find_best_answer_for_passage
from src.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['train', 'validate', 'predict'], default='train')
    arg('--run_root', default='.')
    arg('--batch-size', type=int, default=16)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=0)
    arg('--lr', type=float, default=0.00003)
    arg('--adam_epsilon', type=float, default=1e-8)
    arg('--weight_decay', type=float, default=0.0)
    arg('--fold', type=int, default=0)
    arg('--warmup', type=float, default=0.05)
    arg('--limit', type=int)
    arg('--patience', type=int, default=1)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=20)
    arg('--vocab-size', type=int, default=13318)
    arg('--multi-gpu', type=int, default=0)
    arg('--print-num', type=int, default=5)
    arg('--temperature', type=float)
    
    args = parser.parse_args()

    df = pd.read_table('data/dialog-rewrite/corpus.txt', sep="\t\t", names=['a','b','current','label'], dtype=str)
    df.dropna(how='any', inplace=True)
    train_length = int(len(df)*0.9)
    
    train_df = df.iloc[:train_length].iloc[:, :]
    valid_df = df.iloc[train_length:]
    print(valid_df.head())
    if args.mode == 'predict':
        # valid_df['current'] = valid_df['label']
        valid_df = pd.read_table('data/dialog-rewrite/test.csv', sep=",", names=['a','b','current','label'], dtype=str)
        print(valid_df.tail())
    valid_df['eval_label'] = valid_df['label'].apply(lambda x: ' '.join(list(x)))

    if args.limit:
        train_df = train_df.iloc[0:args.limit]
        valid_df = valid_df.iloc[0:args.limit]
    # train_df['len'] = train_df['content'].apply(lambda x: len(x))

    run_root = Path('experiments/' + args.run_root)
    tokenizer = BertTokenizer.from_pretrained("rbt3")
    valid_set = TaggerRewriterDataset(valid_df, tokenizer, valid=True)
    valid_index = np.array(valid_set.valid_index)
    np.save('index.npy', valid_index)
    valid_df = valid_df.reset_index().loc[valid_index, :]
    ner_index = np.array(valid_set.label_type) == 1
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              collate_fn=tagger_collate_fn)

    config = BertConfig.from_json_file('rbt3/config.json')
    config.num_labels = 5
    # # config.is_decoder = True
    # decoder = BertModel.from_pretrained("../rbt3", config=config)
    # encoder = BertModel.from_pretrained("../rbt3")
    # args.vocab_size = config.vocab_size

    model = TaggerRewriteModel(config)
    model.cuda()

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_set = TaggerRewriterDataset(train_df, tokenizer)

        # np.save('index.npy', train_set.valid_index)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, collate_fn=tagger_collate_fn)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr, eps=args.adam_epsilon)
        t_total = int(len(train_df)*args.n_epochs/args.batch_size)
        warmup_steps = int(t_total*args.warmup)
        # scheduler = get_linear_schedule_with_warmup(
            # optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        # )
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2', verbosity=0)

        train(args, model, optimizer, scheduler, tokenizer,ner_index,
              train_loader=train_loader, valid_df=valid_df, valid_loader=valid_loader,
              epoch_length=len(train_df))

    elif args.mode == 'validate':
        model_path = run_root / ('tagger_model-%d.pt' % args.fold)
        load_model(model, model_path)
        valid_metrics = validate(model, valid_loader, valid_df, args, tokenizer, ner_index, decode_mode='beam_search')
    
    elif args.mode == 'predict':
        model_path = run_root / ('tagger_model-%d.pt' % args.fold)
        load_model(model, model_path)
        valid_metrics = validate(model, valid_loader, valid_df, args, tokenizer, decode_mode='beam_search')


def train(args, model, optimizer, scheduler, tokenizer,ner_index, *,
          train_loader, valid_df, valid_loader, epoch_length,
          n_epochs=None):
    n_epochs = n_epochs or args.n_epochs

    run_root = Path('experiments/' + args.run_root)
    model_path = run_root / ('tagger_model-%d.pt' % args.fold)
    best_model_path = run_root / ('best-model-%d.pt' % args.fold)
    if best_model_path.exists():
        state, best_valid_score = load_model(model, best_model_path)
        start_epoch = state['epoch']
        best_epoch = start_epoch
    else:
        best_valid_score = 0
        start_epoch = 0
        best_epoch = 0
    step = 0
    criterion = CrossEntropyLoss().cuda()
    report_each = 10000
    log = run_root.joinpath('train-%d.log' %
                            args.fold).open('at', encoding='utf8')

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()

        tq = tqdm.tqdm(total=epoch_length)
        losses = []

        mean_loss = 0
        device = torch.device("cuda", 0)
        for i, (ori_sen, token, token_type, start, end, insert_pos, start_ner, end_ner) in enumerate(train_loader):
            input_mask = (token > 0).to(device)
            token, input_mask, token_type, start, end, insert_pos, start_ner, end_ner = \
                token.to(device), input_mask.to(device), token_type.to(device), start.to(
                    device), end.to(device), insert_pos.to(device), start_ner.to(device), end_ner.to(device)
            outputs = model(input_ids=token, attention_mask=input_mask, token_type_ids=token_type,
                            start=start, end=end, insert_pos=insert_pos, start_ner=start_ner,
                            end_ner=end_ner)

            loss = outputs[0]
            if (i + 1) % args.step == 0:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            else:
                with amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

            tq.update(args.batch_size)
            losses.append(loss.item() * args.step)
            mean_loss = np.mean(losses[-report_each:])
            tq.set_postfix(loss=f'{mean_loss:.5f}')
            lr = get_learning_rate(optimizer)
            tq.set_description(f'Epoch {epoch}, lr {lr:.6f}')
            if i and i % report_each == 0:
                write_event(log, step, loss=mean_loss)
            # break
        write_event(log, step, epoch=epoch, loss=mean_loss)
        tq.close()

        valid_metrics = validate(model, valid_loader, valid_df, args, tokenizer, ner_index)
        # write_event(log, step, **valid_metrics)
        current_score = valid_metrics['rouge-1']['f']
        if current_score>best_valid_score:
            print('save success')
            save_model(model, epoch, step, mean_loss, model_path)
            best_valid_score = current_score
    return True


def predict(model, valid_loader, args, tokenizer, progress=False, limit=None, decode_mode='greedy'):
    model.eval()
    all_outputs = []
    if progress:
        if limit is None:
            tq = tqdm.tqdm(total=len(valid_loader))
        else:
            tq = tqdm.tqdm(total=limit)
    device = torch.device("cuda", 0)
    with torch.no_grad():
        for i, (ori_sen, token, token_type, start, end, insert_pos, start_ner, end_ner) in enumerate(valid_loader):
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
    if progress:
        tq.close()
    return all_outputs


def validate(model, valid_loader, valid_df, args, tokenizer, ner_index, save_result=False, progress=False, limit=None,
             decode_mode='greedy'):
    run_root = Path('../experiments/' + args.run_root)
    predictions = predict(model, valid_loader, args, tokenizer, progress=True, limit=limit, decode_mode=decode_mode)
    # valid_df = valid_df.loc[ner_index,:]
    # new_predictions = []
    # for index, item in enumerate(ner_index):
    #     if ner_index[index]:
    #         new_predictions.append(predictions[index])
    # predictions = new_predictions
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


if __name__ == '__main__':
    main()
# seq2seq {'rouge-1': {'f': 0.9060760998445965, 'p': 0.9358447287583956, 'r': 0.8919995514249914}, 'rouge-2': {'f': 0.836360317684148, 'p': 0.8642009100784366, 'r': 0.8245842565193153}, 'rouge-l': {'f': 0.8978019069952512, 'p': 0.9341465162177004, 'r': 0.8772113361415852}, 'em': 0.5315}
# 是否需要改写 {'rouge-1': {'f': 0.9207800706631117, 'p': 0.8989021635835469, 'r': 0.9567351542520496}, 'rouge-2': {'f': 0.8928183325405308, 'p': 0.8696896376814124, 'r': 0.9327111153640774}, 'rouge-l': {'f': 0.9554940232435956, 'p': 0.9594988869180364, 'r': 0.9579031790033299}, 'em': 0.5765}
