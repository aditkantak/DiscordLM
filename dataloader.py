import torch
from torch.utils.data import Dataset, DataLoader
import numpy.typing as npt
import numpy as np
from tokenizer import Tokenizer, VOCAB_FILE, DATA_FILE

class TokenDataset (Dataset):
    def __init__(self, data: npt.NDArray, sequence_length: int):
        super().__init__()
        
        self._data = torch.as_tensor(data, dtype=torch.uint16)
        self._sequence_length = sequence_length

        self.shape = self._data.shape #for testing

    def __len__(self):
        return self._data.shape[0] - self._sequence_length - 1
    
    def __getitem__(self, index):
        example = self._data[index: index + self._sequence_length]
        target = self._data[index + 1: index + self._sequence_length + 1]
        return example, target

if __name__ == "__main__":
    #tokenize and prepare data
    tk = Tokenizer(VOCAB_FILE, 64)
    data = tk.tokenize(DATA_FILE)
    print(f"tokenized data of shape {data.shape}")

    #dataloader test
    dataset = TokenDataset(data, 64)
    dl = DataLoader(dataset, 8, True)
    
    train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    train_loader = DataLoader(train_data, 8, True)

    # # print(f"shape of training data: {train_data.shape}")
    # # print(f"length of test data")

    # for batch_ind, (X, y) in enumerate(train_loader):
    #     print("--------------------------------")
    #     print(f"batch #{batch_ind} with X of shape {X.shape} and y of shape {y.shape}")
    #     print("--------------------------------")
    #     print("".join([tk.decode(i.item()) for i in X[0]]))
    #     print("--------------------------------")
    #     if batch_ind > 1:
    #         break