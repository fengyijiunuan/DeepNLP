import jieba
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from collections import Counter
import os
import re
from tqdm import tqdm
import sys

os.chdir(sys.path[0])

def preprocess_text(text):
    """Preprocess the text by removing unwanted characters and symbols."""
    text = re.sub('----〖新语丝电子文库\(www.xys.org\)〗', '', text)
    text = re.sub('本书来自www.cr173.com免费txt小说下载站', '', text)
    text = re.sub('更多更新免费电子书请关注www.cr173.com', '', text)
    text = re.sub('\u3000', '', text)
    text = re.sub(r'[。，、；：？！（）《》【】“”‘’…—\-,.:;?!\[\](){}\'"<>]', '', text)
    text = re.sub(r'[\n\r\t]', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    return text

def load_data(directory, limit=4):
    """Load data from the specified directory, limited to a certain number of files."""
    corpus = []
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='ansi') as file:
                corpus.append(file.read())
        if i + 1 == limit:
            break
    return corpus

directory = r'C:\Users\86157\Desktop\项目\深度学习与自然语言处理作业\jyxstxtqj_downcc.com'
corpus = load_data(directory)

words = [word for text in corpus for word in jieba.lcut(text)]
print(len(words))

counter = Counter(words)
counter['<unk>'] = 0
tokenizer = get_tokenizer('basic_english')
vocab = Vocab(counter)
vocab_size = len(vocab)

words_str = ' '.join(words)
tokens = tokenizer(words_str)
sequences = [vocab[token] for token in tokens]
sequences = [word if word < vocab_size else vocab['<unk>'] for word in sequences]

import torch
from torch import nn

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.encoder(x)
        out, _ = self.decoder(x, (h, c))
        out = self.fc(out)
        return out

embedding_dim = 256
hidden_units = 50
model = Seq2Seq(vocab_size, embedding_dim, hidden_units)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

assert max(sequences) < vocab_size, "Vocabulary size is insufficient to encode all words."

for epoch in tqdm(range(10)):
    optimizer.zero_grad()
    output = model(torch.tensor(sequences[:1000]))
    loss = criterion(output, torch.tensor(sequences[:1000]))
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model.pth')

model = Seq2Seq(vocab_size, embedding_dim, hidden_units)
model.load_state_dict(torch.load('model.pth'))
model.eval()

start_text = "扬州城中说书先生说到"
start_words = list(jieba.cut(start_text))

word2idx = {word: idx for idx, word in enumerate(counter)}
idx2word = {idx: word for idx, word in enumerate(counter)}

start_sequence = [word2idx[word] for word in start_words if word in word2idx]
if not start_sequence:
    raise ValueError("Start sequence is empty. Please provide a non-empty start sequence.")
input = torch.tensor(start_sequence).long().unsqueeze(0)

max_length = 50
generated_sequence = []

for _ in range(max_length):
    output = model(input)
    output_mean = output.mean(dim=1)
    next_word_idx = output_mean.argmax().item()
    generated_sequence.append(next_word_idx)
    input = torch.tensor([next_word_idx]).unsqueeze(0)

generated_words = [idx2word[idx] for idx in generated_sequence]
generated_text = ''.join(generated_words)

print(generated_text)
