from transformers import BertModel
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF

class BertNER(nn.Module):
    def __init__(self,config):
        super(BertNER, self).__init__()
        self.bert=BertModel.from_pretrained('bert-base-chinese')
        self.num_labels=config.num_labels
        self.dropout=nn.Dropout(config.dropout)
        self.bilstm=nn.LSTM(input_size=768,hidden_size=128,batch_first=True,num_layers=2,bidirectional=True)
        self.classifier=nn.Linear(256,self.num_labels)
        self.crf=CRF(self.num_labels,batch_first=True)
    def forward(self,input_data,token_type_ids=None, attention_mask=None, labels=None):
        input_ids, input_token_starts = input_data
        outputs=self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        padded_sequence_output=pad_sequence(origin_sequence_output,batch_first=True)
        padded_sequence_output=self.dropout(padded_sequence_output)
        lstm_out,_=self.bilstm(padded_sequence_output)
        logits=self.classifier(lstm_out)
        outputs=(logits,)
        if labels is not  None:
            loss_mask=labels.gt(-1)
            loss=self.crf(logits,labels,loss_mask)*(-1)
            outputs=(loss,)+outputs
        return outputs
