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

@dataclass
class Config:
    vocab_size: int
    batch_size: int
    d_model: int
    context_length: int
    num_heads: int
    d_key: int

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._num_heads = config.num_heads
        self._d_key = config.d_key

        self.Wk = nn.Parameter(torch.randn(config.num_heads, config.d_model, config.d_key))
        self.Wq = nn.Parameter(torch.randn(config.num_heads, config.d_model, config.d_key))
        self.Wv = nn.Parameter(torch.randn(config.num_heads, config.d_model, config.d_key))

        mask = torch.tril(torch.ones(config.num_heads, config.context_length, config.context_length)).bool()
        self.register_buffer("mask", mask)

        self.softmax = nn.Softmax(-1)

        self.linear = nn.Linear(config.num_heads * config.d_key, config.num_heads * config.d_key)
        
    def forward(self, X):
        queries = torch.einsum("btc, nck->bntk", *(X, self.Wq))
        keys = torch.einsum("btc, nck->bntk", *(X, self.Wk))
        values = torch.einsum("btc, nck->bntk", *(X, self.Wv))

        attention_scores = queries @ torch.transpose(keys, -1, -2) / math.sqrt(self._d_key)
        attention_scores = attention_scores.masked_fill(self.mask, float("-inf"))
        attention_scores = self.softmax(attention_scores)
        
        logits = torch.einsum("bntt, bntk->bntk", *(attention_scores, values))
        logits = torch.cat([logits[:, i, :, :] for i in range(self._num_heads)], dim = -1)

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

  def forward(self, X):
    X = X + self.attention(self.layernorm_1(X)) #residual connection with prenorm
    X = X + self.feedforward(self.layernorm_2(X)) #residual connection with prenorm
    return X


if __name__ == "__main__":
    config = Config(VOCAB_SIZE, BATCH_SIZE, D_MODEL, CONTEXT_LENGTH, NUM_HEADS, D_KEY)



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
