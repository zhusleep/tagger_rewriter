import os
os.environ["TF_KERAS"] = "1"
from bert4keras.backend import keras, K, set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.layers import Loss
from keras.layers import Input, Lambda, Dense

__all__ = ["taggerRewriterModel"]

set_gelu('tanh')  # 切换gelu版本


class PointerLoss(Loss):
    """定义 total loss 是所有指针 loss 之和，都是多分类交叉熵
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
        total_loss = (start_loss+end_loss+insert_pos_loss+start_ner_loss+end_ner_loss) / 5
        return total_loss


def taggerRewriterModel(model_name, config_path, checkpoint_path, num_classes=5, learning_rate=3e-5):
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
        optimizer=AdamLR(learning_rate=learning_rate, lr_schedule={
            1000: 1,
            2000: 0.1
        }),
        metrics=None,
    )
    return model


if __name__ == '__main__':
    base_dir = '../'
    pretrained_model_dir = f'{base_dir}albert_small_zh_google/'
    config_path = f'{pretrained_model_dir}albert_config_small_google.json'
    checkpoint_path = f'{pretrained_model_dir}albert_model.ckpt'
    dict_path = f'{pretrained_model_dir}vocab.txt'
    model = taggerRewriterModel(model_name='albert', config_path=config_path, checkpoint_path=checkpoint_path)
