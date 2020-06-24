from torch import nn
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from transformers.utils_encoder_decoder import prepare_encoder_decoder_model_kwargs
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class TaggerRewriteModel(nn.Module):
    def __init__(self, config, bert_path):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained(bert_path)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # self.rewrite_output = nn.Linear(config.hidden_size, 2)
        # self.insert_output = F.sigmoid(nn.Linear(config.hidden_size, 1))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start=None,
        end=None,
        insert_pos=None,
        start_ner=None,
        end_ner=None,
        target=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForQuestionAnswering
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        assert answer == "a nice puppet"

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        # pooled_output = outputs[1]
        # rewrite_or_not = self.rewrite_output(pooled_output)
        # pool_loss_fn = nn.BCEWithLogitsLoss()
        # if target[0] is not None:
        #     pool_loss = pool_loss_fn(rewrite_or_not, target)


        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits, insert_pos_logits, start_ner_logits, end_ner_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        insert_pos_logits, start_ner_logits, end_ner_logits = insert_pos_logits.squeeze(-1),\
                                                              start_ner_logits.squeeze(-1), \
                                                              end_ner_logits.squeeze(-1)

        outputs = (start_logits, end_logits, insert_pos_logits, start_ner_logits, end_ner_logits)
        if start is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start.size()) > 1:
                start_positions = start.squeeze(-1)
            if len(end.size()) > 1:
                end_positions = end.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start.clamp_(0, ignored_index)
            end.clamp_(0, ignored_index)
            insert_pos.clamp_(0, ignored_index)
            start_ner.clamp_(0, ignored_index)
            end_ner.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start)
            end_loss = loss_fct(end_logits, end)
            insert_loss = loss_fct(insert_pos_logits, insert_pos)
            start_ner_loss = loss_fct(start_ner_logits, start_ner)
            end_ner_loss = loss_fct(end_ner_logits, end_ner)
            total_loss = (start_loss + end_loss + insert_loss+start_ner_loss+end_ner_loss) / 5
            outputs = (total_loss,) + outputs
            return outputs
        else:
            return (None,) + outputs
            # (loss), start_logits, end_logits, (hidden_states), (attentions)

