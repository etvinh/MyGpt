#taken from Andrej Karpathy on youtube (really good tutorial!) https://www.youtube.com/watch?v=kCc8FmEb1nY

import torch

#toy example
torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
x.shape

xbow = torch.zeros((B,T,C)) #xbow (bag of words)
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C) previous tokens t = all past tokens, C = all 2d information from tokens
        xbow[b,t] = torch.mean(xprev, 0) #average out t

weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(1, keepdim=True)

xbow2 = weights @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)

tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T)) #weights begin with 0, wei tells us how much of each token do we want to aggregate
wei = wei.masked_fill(tril == 0, float('-inf')) #masking tokens from the past (future cannot communicate with the past)
wei = F.softmax(wei, dim=-1) #normalize
xbow3 = wei @ x #aggregation through matrix multiplication
torch.allclose(xbow, xbow3)

