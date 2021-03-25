import argparse
import os
os.environ["TF_KERAS"] = "1"
import pandas as pd
from bert4keras.backend import keras
from dataset import generate_label, data_generator
from bert4keras.tokenizers import Tokenizer
from model import taggerRewriterModel
from sklearn.model_selection import train_test_split
from pointer_decoder import validate
from utils import metrics_fn

parser = argparse.ArgumentParser(__doc__)

parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--n-epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--print-num', type=int, default=5)

args = parser.parse_args()

num_classes = 5
batch_size = args.batch_size
epochs = args.n_epochs
lr = args.lr
valid_sample_limit = args.print_num

base_dir = '../'
pretrained_model_dir = f'{base_dir}albert_small_zh_google/'
config_path = f'{pretrained_model_dir}albert_config_small_google.json'
checkpoint_path = f'{pretrained_model_dir}albert_model.ckpt'
dict_path = f'{pretrained_model_dir}vocab.txt'

data_path = f'{base_dir}data/dialog-rewrite/corpus.txt'

model_weights_save_dir = f'{base_dir}tf_experiments/'
os.makedirs(model_weights_save_dir, exist_ok=True)
model_serve_save_dir = f'{base_dir}tf_rewriter_model/'

tokenizer = Tokenizer(dict_path, do_lower_case=True)

df = pd.read_table(data_path, sep="\t\t", names=['a','b','current','label'], dtype=str, engine='python')
df.dropna(how='any', inplace=True)
train_length = int(len(df)*0.9)
train_df = df.iloc[:train_length].iloc[:, :]
valid_df = df.iloc[train_length:]
#train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df['eval_label'] = valid_df['label'].apply(lambda x: ' '.join(list(x)))
# 加载数据集
train_data = generate_label(train_df, tokenizer)
valid_data = generate_label(valid_df, tokenizer, is_valid=True)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

model = taggerRewriterModel(
    model_name='albert', 
    config_path=config_path, 
    checkpoint_path=checkpoint_path, 
    num_classes=num_classes,
    learning_rate=lr)


class Evaluator(keras.callbacks.Callback):
    """metrics and save best model
    """

    def __init__(self):
        self.best_em = 0.

    def on_epoch_end(self, epoch, logs=None):
        valid_metric = validate(model, valid_generator, valid_df, metrics_fn, example_limit=valid_sample_limit)
        em = valid_metric['em']
        if em > self.best_em:
            self.best_em = em
            model.save_weights(f'{model_weights_save_dir}best_model.weights')
            # model.save(base_dir + '/albert_small_rewriter/2', save_format='tf')
        print(f"best em: {self.best_em:.4f}")


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator],
    )
