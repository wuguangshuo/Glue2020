import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_dir='./log/train.log'
val_split_size=0.2
data_dir='./data/'
files = ['train', 'test']
vocab_path=data_dir + 'vocab.npz'

train_dir ='./data/'  + 'train.npz'
test_dir = './data/' + 'test.npz'


batch_size=2
dropout=0.2
epoch_num=10
lr = 3e-5
model_dir = './save/'+ 'model' + '.pkl'
load_before=True
clip_grad=5
patience = 0.0002
min_epoch_num=2
patience_num=5
labels = ['address', 'book', 'company', 'game', 'government',
          'movie', 'name', 'organization', 'position', 'scene']
ent_type_size=len(labels)
label2id = {
    "address": 0,
    "book": 1,
    "company": 2,
    'game': 3,
    'government': 4,
    'movie': 5,
    'name': 6,
    'organization': 7,
    'position': 8,
    'scene': 9,
}

id2label = {_id: _label for _label, _id in list(label2id.items())}