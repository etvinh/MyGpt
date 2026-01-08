import torch




with open('input.txt', 'r', encoding='utf-8') as f: #open input.txt
    text = f.read()

chars = sorted(list(set(text))) #get set of characters from dataset
vocab = len(chars) #length of char set


#map characters to integers and integers to characters
stoi = {ch:i for i, ch in enumerate(chars)} #iterate over all characters and create lookup table characters to ints
istoi = {i:ch for i, ch in enumerate(chars)} #lookup ints to characters
encode = lambda s: [stoi[c] for c in s] #translate characters to ints individually
decode = lambda l: ''.join([itos[i] for i in l]) #translate ints to characters individually

data = torch.tensor(encode(text), dtype = torch.long) #encode entire dataset

n = int(0.9*len(data)) #use first 90% for train data and last 10% to evaulation data
train_data = data[:n]
val_data = data[n:]

"""block_size = 8 
train_data[:block_size+1]
tensor = ([18, 47, 56, 57, 58, 1, 15, 47, 58])

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")"""

torch.manual_seed(1337)
batch_size = 4 # number of independent sequences proccessed in parallel
block_size = 8 #max context length for predictions

def get_batch(split): #generate small batch of data with input x and target y
    data = train_data if split == 'train' else val_data #assign data to train or val
    ix = torch.randint(len(data) - block_size, (batch_size,))  #create random offset into training set
    x = torch.stack([data[i:i+block_size] for i in ix]) #first block_size characters starting at i
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #x offset by one
    return x, y
xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

"""for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")"""
       
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss