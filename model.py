from torch import nn, tensor
import torch
from tokenizer import Tokenizer, DATA_FILE
import math
from dataclasses import dataclass

VOCAB_SIZE = 64
BATCH_SIZE = 4
D_MODEL = 32
CONTEXT_LENGTH = 8
NUM_HEADS = 4
D_KEY = 8
NUM_BLOCKS = 4

@dataclass
class Config:
    vocab_size: int
    batch_size: int
    d_model: int
    context_length: int
    num_heads: int
    d_key: int
    num_blocks: int

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._num_heads = config.num_heads
        self._d_key = config.d_key

        self.Wqkv = nn.Linear(config.d_model, 3*config.num_heads*config.d_key)

        mask = torch.tril(torch.ones(config.num_heads, config.context_length, config.context_length))
        self.register_buffer("mask", mask)

        self.softmax = nn.Softmax(-1)

        self.linear = nn.Linear(config.num_heads * config.d_key, config.d_model)
        
    def forward(self, X):
        B, T, _ = X.shape

        #B, T, 3*numheads*dkey -> B, T, 3, numheads, dkey -> 3, B, numheads, T, dkey -> 3 x [B, numheads, T, dkey]
        queries, keys, values = self.Wqkv(X).view(B, T, 3, self._num_heads, self._d_key).permute(2, 0, 3, 1, 4)

        #B, numheads, T, dkey @ B, T, dkey, T -> B, numheads, T, T
        attention_scores = queries @ keys.transpose(-1, -2) / math.sqrt(self._d_key)
        #B, numheads, T, T -> B, numheads, T, T (just masking operation)
        attention_scores = attention_scores.masked_fill(self.mask[:, :T, :T] == 0, float("-inf"))
        #B, numheads, T, T -> B, numheads, T, T (probability distribution)
        attention_scores = self.softmax(attention_scores)
        
        #B, numheads, T, T @ B, numheads, T, dkey -> B, numheads, T, dkey
        logits = attention_scores @ values
        #B, numheads, T, dkey -> B, T, numheads, dkey -> B, T, numheads*dkey
        logits = logits.transpose(1, 2).contiguous().view(B, T, self._num_heads * self._d_key)

        #B, T, numheads*dkey -> B, T, dmodel = B, T, C
        logits = self.linear(logits)

        return logits
        
class FeedForward(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.stack = nn.Sequential(nn.Linear(config.d_model, 4 * config.d_model),
                                   nn.ReLU(),
                                   nn.Linear(4 * config.d_model, config.d_model))
        
    def forward(self, X):
        return self.stack(X)
    
class TransformerBlock(nn.Module):
  def __init__(self, config: Config) -> None:
      super().__init__()
      self.attention = MultiHeadSelfAttention(config)
      self.feedforward = FeedForward(config)
      self.layernorm_1 = nn.LayerNorm(config.d_model)
      self.layernorm_2 = nn.LayerNorm(config.d_model)
      self.dropout = nn.Dropout()

  def forward(self, X):
    atten = self.dropout(self.attention(self.layernorm_1(X)))
    X = X + atten #residual connection with prenorm
    ffn = self.dropout(self.feedforward(self.layernorm_2(X)))
    X = X + ffn #residual connection with prenorm
    return X

class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._context_len = config.context_length

        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout()

        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.num_blocks)))
        self.linear = nn.Linear(config.d_model, config.vocab_size)

        self.register_buffer("positional_encodings", self._init_positional_encodings(config))

        self.loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _init_positional_encodings(self, config: Config):
        pe = torch.zeros(config.context_length, config.d_model)
        
        pos = torch.arange(config.context_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2) * math.log(10000) / config.d_model)

        pe[:, ::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)

        return pe.unsqueeze(0)
    
    def forward(self, X, y = None):
        # B, T -> B, T, dmodel
        X = self.embeddings(X)
        # B, T, C -> B, T, C
        X = X + self.positional_encodings[:, :X.shape[1]]
        X = self.dropout(X)
        # B, T, C -> B, T, C
        X = self.blocks(X)
        # B, T, C = B, T, vocab_size
        logits = self.linear(X)

        if y is not None:
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
        else:
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, max_tokens: int, start_seq = None, device = "cpu"):
        seq = start_seq if start_seq is not None else [0]

        for _ in range(max_tokens):
            input_tensor = torch.tensor(seq[-self._context_len:], device = device).unsqueeze(0)
            
            logits, _ = self(input_tensor)
            probs = torch.softmax(logits, -1)[0, -1] #get probabilities for last token

            next_token = torch.multinomial(probs, 1).item() #sample from probs to get next token
            
            seq.append(int(next_token))
            yield next_token

if __name__ == "__main__":
    config = Config(VOCAB_SIZE, BATCH_SIZE, D_MODEL, CONTEXT_LENGTH, NUM_HEADS, D_KEY, NUM_BLOCKS)



"""
        x = (B, T, d_model)

        k = (nHead, d_model, d_key), x * k = (B, nHead, T, d_key) 
        q = (nHead, d_model, d_key), x * q = (B, nHead, T, d_key)
        v = (nHead, d_model, d_key), x * v = (B, nHead, T, d_key)

        keys . queries = (B, nHead, T, T) = scores
        softmax scores

        scores * values = (B, nHead, T, d_key)

        logits = concat along nHead dimension to (B, T, d_key*nHead) = (B, T, C)
        linear C, C*4, relu (or gelu)
        linear C*4, C
"""
