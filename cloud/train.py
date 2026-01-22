import torch
from torch.utils.data import Dataset, DataLoader, random_split #for dataset
from torch import nn #for model
import math #for sqrt in attention
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import requests
import subprocess


print(*(f"{file}\n" for file in Path(".").iterdir()))


DATA_DIR = Path(".")

class IncompletePipelineError (Exception):
    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)

    def __str__(self):
        if not self.message: return f"Data pipeline incomplete. Please run clean_data.py"
        else: return f"Data pipeline incomplete. {self.message} file cannot be found/processed. Please run clean_data.py using \"uv run clean_data.py\""


class Tokenizer:
    def __init__(self, vocab_file: Path, token_dim: int, dump_to_txt = False):
        """
        Creates a tokenizer object with the given `token_dim` token dimension. Requires the dataset as txt file, and a vocab txt file.

        :param self: Description
        :param file_path: data txt file
        :param token_dim: token dimension (vocab_size)
        """
        self.vocab = self.read_vocab(token_dim, vocab_file)

        self.encode_dict = {letter: value for value, letter in enumerate(self.vocab)}
        self.decode_dict = {value: letter for value, letter in enumerate(self.vocab)}

        self.vocab_size = token_dim

    def encode(self, token: str) -> int:
        """
        Encodes the token from a string to a token(int)

        :param self: Description
        :param token: character to encode
        :return: the token representation of the character
        :rtype: int
        """
        return self.encode_dict[token]

    def decode(self, token) -> str:
        """
        Decodes the token from int to its string representation

        :param self: Description
        :param token: token to decode
        :return: the string representation of the token
        :rtype: str
        """
        return self.decode_dict[token]

    def read_vocab(self, vocab_size: int, vocab_file: Path) -> list[str]:
        """
        Extracts the `vocab_size` most common tokens from the data. If vocab_size is greater than the number of unique tokens in the file, returns all unique tokens.

        :param vocab_size: Size of vocabulary to extract
        :type vocab_size: int
        """

        if (not vocab_file.exists()): raise IncompletePipelineError("Vocabulary")

        vocab = []
        i = 0

        with open(vocab_file, "r", encoding="utf-8") as file:
            for line in file:
                if (i >= vocab_size): break
                vocab.append(line[0])
                i += 1

        return vocab

    def tokenize(self, text_file: Path, dump_to_txt = False) -> npt.NDArray:
        """
        Character-tokenizes the txt file at `file_path` into `vocab_size` tokens and saves them to ./data/tokens.npy as an np array, and also saves to txt file if `dump_to_txt` set to true. Returns the array of tokens.

        :param file_path: the file to read the text from
        :param vocab_size: the number of unique characters to count; token values will be within [0 - vocab_size)
        :param dump_to_txt: set to True to dump tokens to txt file also
        :return: the np array of all the tokens
        """
        if not text_file.exists():
            raise IncompletePipelineError("Cleaned messages")

        tokens_file = DATA_DIR / f"{text_file.stem}_data.npy"
        tokens_txt_file = DATA_DIR / f"{text_file.stem}_data.txt"

        if (not Path.exists(tokens_file)):
            tokens = []

            with open(text_file, "r", encoding="utf-8") as data_file:
                    print("Tokenizing text file...")
                    for line in data_file:
                        for char in line:
                            if (char in self.encode_dict):
                                tokens.append(self.encode(char))

            tokens_arr = np.array(tokens, dtype=np.uint16)
            np.save(tokens_file, tokens_arr)
            print(f"Tokenized file saved to {tokens_file}")
        else:
            print(f"Loading tokens from {tokens_file}")
            tokens_arr = np.load(tokens_file)
            print("Tokens loaded")

        if dump_to_txt:
            np.savetxt(tokens_txt_file, tokens_arr, fmt="%.1d")

        return tokens_arr


class TokenDataset (Dataset):
    def __init__(self, data: npt.NDArray, sequence_length: int):
        super().__init__()

        self._data = torch.as_tensor(data, dtype=torch.int64)
        self._sequence_length = sequence_length

        self.shape = self._data.shape #for testing

    def __len__(self):
        return self._data.shape[0] - self._sequence_length - 1

    def __getitem__(self, index):
        example = self._data[index: index + self._sequence_length]
        target = self._data[index + 1: index + self._sequence_length + 1]
        return example, target


@dataclass
class Config:
    vocab_size: int
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
      self.dropout = nn.Dropout(0.1)

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
        self.dropout = nn.Dropout(0.1)

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
    def generate(self, max_tokens: int, device, start_seq = None,):
        seq = start_seq if start_seq is not None else [0]

        for _ in range(max_tokens):
            input_tensor = torch.tensor(seq[-self._context_len:], device = device).unsqueeze(0)

            logits, _ = self(input_tensor)
            probs = torch.softmax(logits, -1)[0, -1] #get probabilities for last token

            next_token = torch.multinomial(probs, 1).item() #sample from probs to get next token

            seq.append(int(next_token))
            yield next_token



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


vocab_file = DATA_DIR / "vocab.txt"
text_file = DATA_DIR / "clean_discord.txt"
tk = Tokenizer(vocab_file, VOCAB_SIZE)
data = tk.tokenize(text_file)
print(f"Initialized data of shape {data.shape}")


dataset = TokenDataset(data, CONTEXT_LENGTH)
print("Loaded dataset")

ttsplit = [TRAIN_TEST_SPLIT, 1-TRAIN_TEST_SPLIT]
train_data, val_data = random_split(dataset, ttsplit)
print("Split dataset")

train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, BATCH_SIZE, shuffle=True)
print(f"Train loader of size {len(train_loader)} initiated")
print(f"Val loader of size {len(val_loader)} initiated")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training process will run on {device}")

model = GPT(CONFIG).to(device)
print(f"Model loaded to {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
print("Training initialization complete")

if not WEIGHTS_DIR.exists(): WEIGHTS_DIR.mkdir()
MODEL_PATH = WEIGHTS_DIR / "best_model.pth"


best_val_loss = float("inf")
    
for epoch in range(EPOCHS):
    #run one epoch
    epoch_running_loss = 0
    lastn_running_loss = 0
    n = 1000

    model.train()

    for batch_num, batch in enumerate(train_loader):
        samples, targets = batch
        samples = samples.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions, loss = model(samples, targets)

        loss.backward()
        optimizer.step()

        epoch_running_loss += loss.item()
        lastn_running_loss += loss.item()

        if (batch_num % n == 0):
            print(f"Batch {batch_num}/{len(train_loader)} ({(batch_num/len(train_loader)):.2f}% epoch completed): Last {n} avg loss: {(lastn_running_loss / n) :.5f} overall avg loss: {(epoch_running_loss / (batch_num + 1)):.5f}")
            lastn_running_loss = 0

    #val loss calculation
    epoch_running_val_loss = 0

    model.eval()
    
    with torch.no_grad():
        for batch_num, batch in enumerate(val_loader):
          samples, targets = batch
          samples = samples.to(device)
          targets = targets.to(device)
    
          predictions, loss = model(samples, targets)
    
          epoch_running_val_loss += loss.item()

    avg_t_loss = epoch_running_loss/len(train_loader)
    avg_v_loss = epoch_running_val_loss/len(val_loader)

    if avg_v_loss < best_val_loss:
        best_val_loss = avg_v_loss
        torch.save(model.state_dict(), MODEL_PATH)
    
    print("-------------------------------------------------------------------------------------------------\n")
    print(f"Epoch {epoch + 1} Complete")
    print(f"Avg training loss: {avg_t_loss:.5f}")
    print(f"Avg val loss:      {avg_v_loss:.5f}")
    print(f"Best val loss:     {best_val_loss:.5f}")
    print("\n-------------------------------------------------------------------------------------------------")

print(f"Training complete. Best val_loss achieved: {best_val_loss:.5f}")

#generate some text 
def generate_text(gpt, max_tokens, start_seq=""):
    tk = Tokenizer(vocab_file, CONFIG.vocab_size)
    output = start_seq

    if start_seq: 
        seq = [tk.encode(c) for c in start_seq]
        print(start_seq, end="")
    else: 
        seq = [0]

    for token in gpt.generate(max_tokens, device, seq):
        next_token = tk.decode(token)
        print(next_token, end="")
        output += next_token

    print()

    return output

model = GPT(CONFIG)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.to(device)
print(f"Generation starting on {device}.")
print()

output = generate_text(model, 200)

#uploading model weights
result = subprocess.run(
    ["curl", "--upload-file", "./weights/best_model.pth", "https://transfer.sh/best_model.pth"],
    capture_output=True, text=True
)
model_file_link = result.stdout
print(f"Model uploaded to \n{model_file_link}\n")

#send it to phone
url = "https://ntfy.sh/discordlm_training"
try:
    requests.post(url, data=f"Model located at: {model_file_link} Generated output: {output}".encode(encoding="utf-8"))
except:
    print("push notif failed")


