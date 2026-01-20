import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model import GPT, Config
from dataloader import TokenDataset
from tokenizer import Tokenizer, VOCAB_FILE, DATA_FILE

VOCAB_SIZE = 64
D_MODEL = 32
CONTEXT_LENGTH = 8
NUM_HEADS = 4
D_KEY = 8
NUM_BLOCKS = 4

BATCH_SIZE = 4
TRAIN_TEST_SPLIT = 0.9
LR = 1e-4
MOMENTUM = 0.9

def train_epoch(loader: DataLoader):
    pass

if __name__ == "__main__":
    tk = Tokenizer(VOCAB_FILE, VOCAB_SIZE)
    data = tk.tokenize(DATA_FILE)
    print(f"Initialized data of shape {data.shape}")

    dataset = TokenDataset(data, CONTEXT_LENGTH)
    print("Loaded dataset")

    ttsplit = [TRAIN_TEST_SPLIT, 1-TRAIN_TEST_SPLIT]
    train_data, val_data = random_split(dataset, ttsplit)
    print("Split dataset")

    loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training process will run on {device}")

    config = Config(VOCAB_SIZE, 
                    #BATCH_SIZE, 
                    D_MODEL, 
                    CONTEXT_LENGTH, 
                    NUM_HEADS, 
                    D_KEY, 
                    NUM_BLOCKS)
    
    model = GPT(config).to(device)
    print(f"Model loaded to {device}")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

