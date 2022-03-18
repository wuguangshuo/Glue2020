
log_dir='./log/train.log'
data_dir='./data/'
files = ['train', 'test']
vocab_path=data_dir + 'vocab.npz'

train_dir ='./data/'  + 'train.npz'
test_dir = './data/' + 'test.npz'
case_dir='./data/'  + 'bad_case.txt'
dev_split_size=0.2
max_vocab_size=1000000
batch_size=64
epoch_num=30
embedding_size=256
hidden_size=128
drop_out=0.2

lr = 0.001
lr_step = 5
lr_gamma = 0.8

min_epoch_num = 5
patience = 0.0002
patience_num = 5

model_dir = './save/' + 'model.pth'
labels = ['address', 'book', 'company', 'game', 'government',
          'movie', 'name', 'organization', 'position', 'scene']

label2id = {
    "O": 0,
    "B-address": 1,
    "B-book": 2,
    "B-company": 3,
    'B-game': 4,
    'B-government': 5,
    'B-movie': 6,
    'B-name': 7,
    'B-organization': 8,
    'B-position': 9,
    'B-scene': 10,
    "I-address": 11,
    "I-book": 12,
    "I-company": 13,
    'I-game': 14,
    'I-government': 15,
    'I-movie': 16,
    'I-name': 17,
    'I-organization': 18,
    'I-position': 19,
    'I-scene': 20,
    "S-address": 21,
    "S-book": 22,
    "S-company": 23,
    'S-game': 24,
    'S-government': 25,
    'S-movie': 26,
    'S-name': 27,
    'S-organization': 28,
    'S-position': 29,
    'S-scene': 30,
}

id2label = {_id: _label for _label, _id in list(label2id.items())}