import torch
import config
from model import BertNER
from transformers import BertTokenizer
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text = input("input:")
test=list(text)
words=[]
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
for word in text:
    words.extend(tokenizer.tokenize(word))
example=tokenizer.convert_tokens_to_ids(words)
n=len(example)
example=torch.LongTensor(example).to(device)

example=example.unsqueeze(0)
input_token_starts=torch.tensor([1]*n).to(device)
input_token_starts=input_token_starts.unsqueeze(0)
example=[example,input_token_starts]
a,b=example
state = torch.load(config.model_dir)
model = BertNER(config)
model.to(device)
model.load_state_dict(state['model_state'])

tmp=model.forward(example)[0]

res = model.crf.decode(tmp)
res=list(itertools.chain.from_iterable(res))
result=[config.id2label[r] for r in res]
print(result)