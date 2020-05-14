import torch
import torch.nn as nn

class XLNet(nn.Module):
    def __init__(self, dropout_rate=0.1, hidden_size=768, num_labels=2):
        super(custom_bert, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(pretrained_model, output_attentions=False, output_hidden_states=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, input_ids,token_type_ids,attention_mask,labels=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs[1])
        logits = self.classifier(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss,logits
        return logits