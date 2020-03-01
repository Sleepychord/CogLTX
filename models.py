import torch
import torch.nn.functional as F
from transformers import BertPreTrainedModel, RobertaConfig, RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP

class Introspector(BertPreTrainedModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(Introspector, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)



class Reasoner(object): # Interface
    def export_labels(self, bufs, device):
        raise NotImplementedError
        # return (labels: consistent with forward, crucials: list of list of blks)
    def forward(self, ids, attn_masks, type_ids, labels, *args, **kwargs):
        raise NotImplementedError
        # return (loss, ) if labels is not None else ...

class QAReasoner(Reasoner, BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(QAReasoner, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def export_labels(self, bufs, device):
        max_len = max([buf.calc_size() for buf in bufs])
        start, end = torch.zeros(2, len(bufs), max_len, dtype=torch.long, device=device)
        for i, buf in enumerate(bufs):
            t = 0
            for b in buf.blocks:
                w = t + len(b)
                if hasattr(b, 'start'):
                    start[i, t + b.start[0]] = b.start[1]
                if hasattr(b, 'end'):
                    end[i, t + b.end[0]] = b.end[1]
                t = w
        return start, end

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if labels is not None:
            start_positions, end_positions = labels
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
