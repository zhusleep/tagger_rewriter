import os
os.environ["TF_KERAS"] = "1"
import tensorflow as tf
import pandas as pd
import numpy as np
from dataset import generate_label, data_generator
from bert4keras.backend import keras, K, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import Loss
from keras.layers import Input, Lambda, Dense
from sklearn.model_selection import train_test_split
from utils import find_best_answer, find_best_answer_for_passage, metrics_fn

set_gelu('tanh')  # 切换gelu版本

num_classes = 5
maxlen = 128
batch_size = 32
lr = 3e-5
base_dir = '../../'
config_path = '../../albert_small_zh_google/albert_config_small_google.json'
checkpoint_path = '../../albert_small_zh_google/albert_model.ckpt'
dict_path = '../../albert_small_zh_google/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)

df = pd.read_table('../../data/dialog-rewrite/corpus.txt', sep="\t\t", names=['a','b','current','label'], dtype=str, engine='python')
df.dropna(how='any', inplace=True)
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df['eval_label'] = valid_df['label'].apply(lambda x: ' '.join(list(x)))
# 加载数据集
train_data = generate_label(train_df, tokenizer)
valid_data = generate_label(valid_df, tokenizer, is_valid=True)
valid_data_acc = generate_label(valid_df, tokenizer)

# 补充输入
start_labels = Input(shape=(1, ), name='Start-Labels')
end_labels = Input(shape=(1, ), name='End-Lables')
insert_pos_labels = Input(shape=(1, ), name='Insert-Pos-Labels')
start_ner_labels = Input(shape=(1, ), name='Start-NER-Labels')
end_ner_labels = Input(shape=(1, ), name='End-NER-Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    return_keras_model=False,
)

output = bert.model.output
output = Dense(
    units=num_classes,
    activation='linear',
    kernel_initializer=bert.initializer
)(output)

start_pred = Lambda(lambda x: x[:, :, 0], name='start')(output)
end_pred = Lambda(lambda x: x[:, :, 1], name='end')(output)
insert_pos_pred = Lambda(lambda x: x[:, :, 2], name='insrt_pos')(output)
start_ner_pred = Lambda(lambda x: x[:, :, 3], name='start_ner')(output)
end_ner_pred = Lambda(lambda x: x[:, :, 4], name='end_ner')(output)


class PointerLoss(Loss):
    """所有指针loss之和，都是多分类交叉熵
    """
    def compute_loss(self, inputs, mask=None):
        start_labels, end_labels, insert_pos_labels, start_ner_labels, end_ner_labels = inputs[:5]
        start_pred, end_pred, insert_pos_pred, start_ner_pred, end_ner_pred = inputs[5:]
        # 各个指针的loss
        start_loss = K.sparse_categorical_crossentropy(start_labels, start_pred, from_logits=True)
        start_loss = K.mean(start_loss)
        end_loss = K.sparse_categorical_crossentropy(end_labels, end_pred, from_logits=True)
        end_loss = K.mean(end_loss)
        insert_pos_loss = K.sparse_categorical_crossentropy(insert_pos_labels, insert_pos_pred, from_logits=True)
        insert_pos_loss = K.mean(insert_pos_loss)
        start_ner_loss = K.sparse_categorical_crossentropy(start_ner_labels, start_ner_pred, from_logits=True)
        start_ner_loss = K.mean(start_ner_loss)
        end_ner_loss = K.sparse_categorical_crossentropy(end_ner_labels, end_ner_pred, from_logits=True)
        end_ner_loss = K.mean(end_ner_loss)
        # 总的loss
        total_loss = (start_loss+end_loss+insert_pos_loss+start_ner_loss+end_ner_loss) / num_classes
        return total_loss

start_pred, end_pred, insert_pos_pred, start_ner_pred, end_ner_pred = PointerLoss([5,6,7,8,9])([start_labels, end_labels, insert_pos_labels, start_ner_labels, end_ner_labels,
start_pred, end_pred, insert_pos_pred, start_ner_pred, end_ner_pred])

model = keras.models.Model(bert.model.inputs + [start_labels, end_labels, insert_pos_labels, start_ner_labels, end_ner_labels], 
[start_pred, end_pred, insert_pos_pred, start_ner_pred, end_ner_pred])
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam)

model.compile(
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=3e-5, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=None,
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
valid_acc_generator = data_generator(valid_data_acc, batch_size)


def evaluate_acc(data):
    total, right = 0., 0.
    for model_inputs, _ in data:
        start_labels, end_labels, insert_pos_labels, start_ner_labels, end_ner_labels = model_inputs[2:]
        start_pred, end_pred, insert_pos_pred, start_ner_pred, end_ner_pred = model(model_inputs)

        start_pred = start_pred.numpy().argmax(axis=1)
        end_pred = end_pred.numpy().argmax(axis=1)
        insert_pos_pred = insert_pos_pred.numpy().argmax(axis=1)
        start_ner_pred = start_ner_pred.numpy().argmax(axis=1)
        end_ner_pred = end_ner_pred.numpy().argmax(axis=1)
        
        start_right = (start_labels == start_pred).sum()
        end_right = (end_labels == end_pred).sum()
        insert_pos_right = (insert_pos_labels == insert_pos_pred).sum()
        start_ner_right = (start_ner_labels == start_ner_pred).sum()
        end_ner_right = (end_ner_labels == end_ner_pred).sum()

        batch_right = min([start_right, end_right, insert_pos_right, start_ner_right, end_ner_right])
        total += len(start_labels)
        right += batch_right
    return right / total


def decode_out(data):
    all_outputs = []
    for model_inputs, ori_sen in data:
        token, token_type, start_labels, end_labels, insert_pos_labels, start_ner_labels, end_ner_labels = model_inputs
        start_logits, end_logits, insert_pos_logits, start_ner_logits, end_ner_logits = model(model_inputs)

        for i in range(len(token)):
            try:
                split_index = len(ori_sen[i][1])+1
                (best_start, best_end), max_prob = find_best_answer_for_passage(start_logits[i], end_logits[i], split_index)
            except:
                print('something error!')
                pass
            token_seq = ''.join(ori_sen[i])
            info_pos = (best_start, best_end)
            token_subseq = token_seq[info_pos[0]:(info_pos[1]+1)]
            text = token_subseq
            # print('关键信息检测 ', text)
            context_len = sum(token_type[i] == 0)
            input_len = len(token_seq)
            if info_pos[1] == 0 or len(text) == 0 or text in token_seq[context_len::]:
                all_outputs.append(token_seq[context_len:input_len-1])
                continue
            # 插入和指代做比较
            insert_prob = max(insert_pos_logits[i].numpy())
            # 指代概率
            (best_start, best_end), max_ner_prob = find_best_answer(start_ner_logits[i], end_ner_logits[i])
            if  best_start>=context_len and max_ner_prob>insert_prob:
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
                continue
            else:# best_start<context_len:
                # 指代消歧
                all_outputs.append(token_seq[context_len:input_len-1])

    # print(all_outputs)
    return all_outputs


def decode_out_new(data, model):
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


def validate(valid_generator, valid_df):
    predictions = decode_out_new(valid_generator, model)

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
        if i >= 5:
            break
    return valid_metric


class Evaluator(keras.callbacks.Callback):

    def __init__(self):
        self.best_em = 0.

    def on_epoch_end(self, epoch, logs=None):
        pointer_acc = evaluate_acc(valid_acc_generator)
        valid_metric = validate(valid_generator, valid_df)
        em = valid_metric['em']
        print(f"pointer acc: {pointer_acc:.5f} em: {em:.5f}\n")
        if em > self.best_em:
            self.best_em = em
            model.save_weights('best_model.weights')
            model.save(base_dir + '/albert_small_rewriter/2', save_format='tf')


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[evaluator],
    )

    #model.load_weights('best_model.weights')
    #print(u'final test acc: %05f\n' % (evaluate(test_generator)))

else:
    model.load_weights('best_model.weights')