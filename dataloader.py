import torch
import numpy.typing as npt
import numpy as np
from tokenizer import Tokenizer, VOCAB_FILE, DATA_FILE

class TokenDataset (torch.utils.data.Dataset):
    def __init__(self, data: npt.NDArray, sequence_length: int):
        super().__init__()
        
        self.data = torch.as_tensor(data, dtype=torch.uint16)
        self.sequence_length = sequence_length

    def __len__(self):
        return self.data.shape[0] - self.sequence_length - 1




class DataLoader:
    def __init__(self, data: npt.NDArray, ttsplit: int):
        self.data = torch.as_tensor(data, dtype=torch.uint16)

        self.dataset_length, = self.data.shape

        self.train_data = self.data[:int(self.dataset_length*ttsplit)]
        self.val_data = self.data[int(self.dataset_length*ttsplit):]

    def get_batch(self, batch_size: int, sequence_length: int, batch_type: str):
        data = self.train_data if batch_type == "train" else self.val_data
        
        offsets = torch.randint(0, data.shape[0] - sequence_length, (batch_size,))
        
        X = torch.stack([data[i: i + sequence_length] for i in offsets])
        y = torch.stack([data[i + 1: i + sequence_length + 1] for i in offsets])
        return X, y

if __name__ == "__main__":
    #tokenize and prepare data
    tk = Tokenizer(VOCAB_FILE, 64)
    data = tk.tokenize(DATA_FILE)
    print(f"tokenized data of shape {data.shape}")

    #dataloader test
    dl = DataLoader(data, 0.9)
    X, y = dl.get_batch(8, 64, "train")
    print(f"shape of X: {X.shape}")
    print(f"shape of y: {y.shape}")