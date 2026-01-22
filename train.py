import torch
from torch.utils.data import Dataset, DataLoader, random_split
from model import GPT, Config
from dataloader import TokenDataset
from tokenizer import Tokenizer, VOCAB_FILE, DATA_FILE
from pathlib import Path
import datetime

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

if __name__ == "__main__":
    tk = Tokenizer(VOCAB_FILE, VOCAB_SIZE)
    data = tk.tokenize(VOCAB_FILE)
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