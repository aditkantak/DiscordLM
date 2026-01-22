import torch
from model import GPT, Config
from pathlib import Path
from tokenizer import Tokenizer, VOCAB_FILE, DATA_FILE
import requests, subprocess

MODEL_PATH = Path("./weights/best_model.pth")

VOCAB_SIZE = 64
D_MODEL = 256
CONTEXT_LENGTH = 128
NUM_HEADS = 8
D_KEY = 32
NUM_BLOCKS = 4

BATCH_SIZE = 256
TRAIN_TEST_SPLIT = 0.9
LR = 3e-4
MOMENTUM = 0.9
EPOCHS = 5
WEIGHT_DECAY = 0.1

WEIGHTS_DIR = Path(".") / "weights"

CONFIG = Config(VOCAB_SIZE,
                D_MODEL,
                CONTEXT_LENGTH,
                NUM_HEADS,
                D_KEY,
                NUM_BLOCKS)

def generate_text(gpt, max_tokens, start_seq=""):
    tk = Tokenizer(VOCAB_FILE, CONFIG.vocab_size)
    output = start_seq

    if start_seq: 
        seq = [tk.encode(c) for c in start_seq]
        print(start_seq, end="")
    else: 
        seq = [0]

    for token in gpt.generate(max_tokens, seq):
        next_token = tk.decode(token)
        print(next_token, end="")
        output += next_token

    print()

    return output

if __name__ == "__main__":
    model = GPT(CONFIG)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.to(device)
    print(f"starting generation on {device}")
    print()
    print()

    output = generate_text(model, 1000)

    url = "https://ntfy.sh/discordlm_training"
    try:
        requests.post(url, data=output.encode(encoding="utf-8"))
        #print(x.text)
    except Exception as e:
        print("push notif failed due to", e)

    # result = subprocess.run(
    # ["curl", "--upload-file", "./weights/best_model.pth", "https://transfer.sh/best_model.pth"],
    # capture_output=True, text=True
    # )
    # print(result)