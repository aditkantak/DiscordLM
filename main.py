import torch
from model import GPT, Config
from pathlib import Path
from tokenizer import Tokenizer, VOCAB_FILE, DATA_FILE
import requests, subprocess
import time

MODEL_PATH = Path("./weights/model_1-08.pth")

VOCAB_SIZE = 64
D_MODEL = 256
CONTEXT_LENGTH = 128
NUM_HEADS = 8
D_KEY = 32
NUM_BLOCKS = 5

TEMPERATURE = 0.7

NUM_TOKENS = 1000

CONFIG = Config(VOCAB_SIZE,
                D_MODEL,
                CONTEXT_LENGTH,
                NUM_HEADS,
                D_KEY,
                NUM_BLOCKS)

#generate some text 
def generate_text(gpt, max_tokens, temperature, device, tokenizer, start_seq=""):
    output = start_seq

    if start_seq: 
        seq = [tokenizer.encode(c) for c in start_seq]
        print(start_seq, end="")
    else: 
        seq = [0]

    for token in gpt.generate(max_tokens, temperature, seq, device=device):
        next_token = tokenizer.decode(token)
        print(next_token, end="")
        output += next_token

    print()

    return output

if __name__ == "__main__":
    tk = Tokenizer(VOCAB_FILE, VOCAB_SIZE)

    model = GPT(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"{sum(m.numel() for m in model.parameters() if m.requires_grad)} trainable parameters")
    print(f"starting generation on {device}\n")

    start = time.perf_counter()

    output = generate_text(model, NUM_TOKENS, TEMPERATURE, device, tk)

    end = time.perf_counter()
    
    print(f"\nTime taken for {NUM_TOKENS} tokens on {device}: {end-start:.5f}")