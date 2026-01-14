import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 4 # number of independent sequences proccessed in parallel
block_size = 8 #max context length for predictions
max_iterations = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' #if you have a GPU use cuda instead of CPU
eval_iterations = 200
n_embed = 32
#---------------------------------------------------------
torch.manual_seed(1337)


with open('input.txt', 'r', encoding='utf-8') as f: #open input.txt
    text = f.read()

chars = sorted(list(set(text))) #get set of characters from dataset
vocab_size = len(chars) #length of char set
#map characters to integers and integers to characters
stoi = {ch:i for i, ch in enumerate(chars)} #iterate over all characters and create lookup table characters to ints
itos = {i:ch for i, ch in enumerate(chars)} #lookup ints to characters
encode = lambda s: [stoi[c] for c in s] #translate characters to ints individually
decode = lambda l: ''.join([itos[i] for i in l]) #translate ints to characters individually

data = torch.tensor(encode(text), dtype = torch.long) #encode entire dataset

n = int(0.9*len(data)) #use first 90% for train data and last 10% to evaulation data
train_data = data[:n]
val_data = data[n:]



def get_batch(split): #generate small batch of data with input x and target y
    data = train_data if split == 'train' else val_data #assign data to train or val
    ix = torch.randint(len(data) - block_size, (batch_size,))  #create random offset into training set
    x = torch.stack([data[i:i+block_size] for i in ix]) #first block_size characters starting at i
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #x offset by one
    x, y = x.to(device), y.to(device) #move data to GPU after its loaded
    return x, y

@torch.no_grad() #tell pytorch to not call .backward on estimate_loss()
def estimate_loss(): #averages loss over multiple batches
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations): #iterate eval_iterations times 
            X, Y = get_batch(split)   #get loss
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() #get average loss from both splits
    model.train()
    return out



class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # create a vocab_size x vocab_size token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.ml_head = nn.Linear(n_embed, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) #each position from 0 to block_size - 1 gets an embed
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx) #every integer in input refers to embedding table and
        #plucks row from embedding table corresponding to index breaking down into Batch = 4, time = 8 and channels = embed (B,T,C)
        #logits are predictions
        pos_embedding = self.position_embedding_table(torch.arange(T), device =device) #creates (T, C) tensor of position embeddings
        x = token_embeddings + pos_embedding #add position embeddings to token embeddings (B,T,C)
        logits = self.lm_head(x) #(B,T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape 
            logits = logits.view(B*T, C) #change C to second dimension to work with Pytorch
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #measures quality of logits with respect to targets

        return logits, loss

    def generate(self, idx, max_new_tokens): #generate function for model take (B+T) and make (B+T) + 1,2,3, etc. until max new †okens
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device) #move model parameters to GPU

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) #create AdamW optimizer lr=learning rate
for iterations in range(max_iterations): # increase number of steps for good results...
    if iterations % eval_interval == 0: 
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}") #report train validation loss
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device) #create context on GPU
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))