from transformers import ElectraPreTrainedModel, ElectraModel
from torch.nn import Dropout, Linear, Softmax
import torch
import torch.nn.functional as F

class ElectraForMultipleChoicePlus(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, 1)
        self.softmax = Softmax(dim=1)
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        
        outputs = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[0][:, 0]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss, reshaped_logits
        else:
            return reshaped_logits

    def predict(self, input_ids, attention_mask=None, token_type_ids=None):
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        predictions = self.softmax(logits)
        return predictions
