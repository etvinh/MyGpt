#taken from Andrej Karpathy on youtube (really good tutorial!) https://www.youtube.com/watch?v=kCc8FmEb1nY
import torch
import torch.nn as nn
from torch.nn import functional as F


#hyperparameters
batch_size = 4 # number of independent sequences proccessed in parallel
block_size = 8 #max context length for predictions
max_iterations = 5000
eval_interval = 300
learning_rate = 1e-3
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

class Head(nn.Module):

#final self-attention implementation

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #lower triangular matrix to mask future tokens
    def forward(self, x):
        B,T,C = x.shape


        #single head to perform self-attention
        k = self.key(x)   #(B,T, head_size)
        q = self.query(x) #(B,T, head_size)
        #all queries dot product with all keys
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B,T, head_size) @ (B, head_size, T) ----> (B,T,T) C is scaling factor to prevent large dot products
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #masking
        wei = F.softmax(wei, dim=-1) #normalize
        v = self.value(x) #(B,T, head_size)
        out = wei @ v #aggregation through matrix multiplication
        return out
class multiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #create multiple heads
        self.proj = nn.Linear(n_embed, n_embed) #final projection layer
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenate outputs of all heads
        out = self.proj(out) #final projection
        return out
    



class Feedforward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), #expand embedding size by 4
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), #project back to n_embed
        )
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.sa_head = multiheadAttention(n_heads, n_embed//n_heads) #4 heads, each head size is n_embed/4
        self.ffwd = Feedforward(n_embed) 
        self.ln1 = nn.LayerNorm(n_embed) #layer norm before self-attention
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x +self.sa_head(self.ln1(x)) #use residual connection 
        x = x+ self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # create a vocab_size x vocab_size token embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, 4),
            Block(n_embed, 4),
            Block(n_embed, 4),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.sa_head = multiheadAttention(4, n_embed//4) #4 heads, each head size is n_embed/4
        self.ffwd = Feedforward(n_embed) 
        self.position_embedding_table = nn.Embedding(block_size, n_embed) #each position from 0 to block_size - 1 gets an embed

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx) #every integer in input refers to embedding table and
        #plucks row from embedding table corresponding to index breaking down into Batch = 4, time = 8 and channels = embed (B,T,C)
        #logits are predictions
        pos_embedding = self.position_embedding_table(torch.arange(T, device =device)) #creates (T, C) tensor of position embeddings
        x = token_embeddings + pos_embedding #add position embeddings to token embeddings (B,T,C)
        x = self.sa_head(x) #apply self-attention head
        x = self.ffwd(x) #apply feedforward network
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
            idx_cond = idx[:, -block_size:] #crop idx to last block_size tokens
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



    
model = BigramLanguageModel()
m = model.to(device) #move model parameters to GPU

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) #create AdamW optimizer lr=learning rate
for iterations in range(max_iterations): # increase number of steps for good results...
    if iterations % eval_interval == 0: 
        losses = estimate_loss()
        print(f"step {iterations}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device) #create context on GPU
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

