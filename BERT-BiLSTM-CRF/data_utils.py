import logging
import numpy as np
import torch
from torch.utils.data import Dataset
import os,json
from transformers import BertTokenizer
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
                labels = ['O'] * len(words)

                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert ''.join(words[start_index:end_index + 1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-' + key
                                else:
                                    labels[start_index] = 'B-' + key
                                    labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("--------{} data process DONE!--------".format(mode))



class NERDataset(Dataset):
    def __init__(self,words,labels,config,word_pad_idx=0,label_pad_idx=-1):
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
        labels=[]

        for line in origin_sentences:
            n=len(line)
            token_start_idxs=[]
            for i in range(n):
                token_start_idxs.extend([i+1])
            token_start_idxs=np.array(token_start_idxs)
            words=['[CLS]']+line
            sentences.append((self.tokenizer.convert_tokens_to_ids(words),token_start_idxs))
        for tag in origin_labels:
            label_id=[self.label2id.get(t) for t in tag]
            labels.append(label_id)
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
        max_len=max([len(s[0]) for s in sentences])
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts=[]
        for j in range(batch_len):
            cur_len=len(sentences[j][0])
            batch_data[j][:cur_len]=sentences[j][0]
        #去除cls和pad对模型的影响
        for j in range(batch_len):
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)

        batch_labels = self.label_pad_idx * np.ones((batch_len, max_len-1))
        for j in range(batch_len):
            cur_tags_len=len(labels[j])
            batch_labels[j][:cur_tags_len]=labels[j]

        batch_data=torch.tensor(batch_data,dtype=torch.long).to(self.device)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long).to(self.device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

        return [batch_data, batch_label_starts, batch_labels]

















