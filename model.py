from torch import nn, tensor
import torch
from tokenizer import Tokenizer, DATA_FILE

VOCAB_SIZE = 64
EMBEDDING_DIM = 128

class Bigram (nn.Module):
    def __init__ (self):
        super().__init__()
        self.embedding_table = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)

    def forward(self, X, target = None):
        logits = self.embedding_table(X)
        return logits

if __name__ == "__main__":
    tokenizer = Tokenizer(64)