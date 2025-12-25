import os
import time

DIRECTORY_PATH = "./raw/Messages/"
DATA_FILE = "./data/clean_discord.txt"
VOCAB_FILE = "./data/vocab.txt"
LOGS_FILE = "./data/logs.txt"
MISC_FILE = "./data/misc.txt"
TOKENIZED_DATA_FILE = "./data/tokenized.txt"

def read_vocab(vocab_size: int) -> list[int]:
    """
    Extracts the `vocab_size` most common tokens from the data. If vocab_size is greater than the number of unique tokens in the file, returns all unique tokens.
    
    :param vocab_size: Size of vocabulary to extract
    :type vocab_size: int
    """
    vocab = []
    i = 0
    with open(VOCAB_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if (i >= vocab_size): break
            vocab.append(line[0])
            i += 1

    return vocab


if __name__ == "__main__":
    if os.path.exists(TOKENIZED_DATA_FILE):
        os.remove(TOKENIZED_DATA_FILE)

    vocab = read_vocab(64)

    vocab_encode = {letter: value for value, letter in enumerate(vocab)}
    vocab_decode = {value: letter for value, letter in enumerate(vocab)}

    with open(DATA_FILE, "r", encoding="utf-8") as data_file:
        with open(TOKENIZED_DATA_FILE, "a", encoding="utf-8") as out_file:
            for line in data_file:
                for char in line:
                    if (char in vocab_encode): out_file.write(f"{vocab_encode[char]} ")

    i = 0
    with open(TOKENIZED_DATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            tokens = line.split()
            for token in tokens:
                print(vocab_decode[int(token)], end="", flush=True)
                time.sleep(0.01)
                i += 1
                if i > 10000: break


