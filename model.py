from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel


class TaggerRewriteModel(nn.Module):
    def __init__(self, config, bert_path):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained("voidful/albert_chinese_tiny")
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_outputs = nn.Linear(312, config.num_labels)

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
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
