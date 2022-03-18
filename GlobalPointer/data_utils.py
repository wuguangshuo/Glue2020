import logging
import numpy as np
import torch
from torch.utils.data import Dataset
import os,json
from transformers import BertTokenizer
import config
class Processor:
    def __init__(self,config):
        self.data_dir=config.data_dir
        self.files=config.files
    def data_process(self):
        for file_name in self.files:
            self.get_examples(file_name)
    def get_examples(self,mode):
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            # 先读取到内存中，然后逐行处理
            for line in f.readlines():
                # loads()：用于处理内存中的json对象，strip去除可能存在的空格
                json_line = json.loads(line.strip())

                text = json_line['text']
                words = list(text)
                # 如果没有label，则返回None
                label_entities = json_line.get('label', None)
                labels = []

                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                labels.append([start_index+1, end_index+1,key])#因为后续会加入CLS

                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("--------{} data process DONE!--------".format(mode))



class NERDataset(Dataset):
    def __init__(self,words,labels,config,word_pad_idx=0,label_pad_idx=0):
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)
        self.label2id=config.label2id
        self.id2label=config.id2label
        self.dataset=self.preprocess(words,labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config.device
    def preprocess(self,origin_sentences,origin_labels):
        data=[]
        sentences=[]
        for line in origin_sentences:
            words=['[CLS]']+line
            sentences.append(self.tokenizer.convert_tokens_to_ids(words))
        labels=origin_labels.tolist()
        for sentence,label in zip(sentences,labels):
            data.append((sentence,label))
        return data
    def __getitem__(self, idx):
        word=self.dataset[idx][0]
        label=self.dataset[idx][1]
        return [word,label]
    def __len__(self):
        return len(self.dataset)

    def collate_fn(self,batch):
        sentences=[x[0] for x in batch]
        labels = [x[1] for x in batch]
        batch_len = len(sentences)
        max_len=max([len(s) for s in sentences])
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))

        for j in range(batch_len):
            cur_len=len(sentences[j])
            batch_data[j][:cur_len]=sentences[j]

        batch_attention_mask = self.word_pad_idx * np.ones((batch_len, max_len))
        for j in range(batch_len):
            cur_len=len(sentences[j])
            batch_attention_mask[j][:cur_len]=1

        batch_token_type = 1 * np.ones((batch_len, max_len))
        for j in range(batch_len):
            cur_len=len(sentences[j])
            batch_token_type[j][:cur_len]=0

        batch_labels = self.label_pad_idx * np.ones((batch_len, config.ent_type_size,max_len,max_len))
        for j in range(batch_len):
            for start, end, label in labels[j]:
               batch_labels[j,config.label2id[label], start, end] = 1

        batch_data=torch.tensor(batch_data,dtype=torch.long).to(self.device)
        batch_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(self.device)
        batch_token_type = torch.tensor(batch_token_type, dtype=torch.long).to(self.device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

        return [batch_data, batch_attention_mask, batch_token_type,batch_labels]

















