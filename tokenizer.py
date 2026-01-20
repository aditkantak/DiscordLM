import sys
from pathlib import Path
import numpy as np
import numpy.typing as npt

BASE_DIR = Path(".")
DIRECTORY_PATH = BASE_DIR / "raw_discord" / "Messages"
DATA_DIR = BASE_DIR / "data"

DATA_FILE = DATA_DIR / "clean_discord.txt"
VOCAB_FILE = DATA_DIR / "vocab.txt"
LOGS_FILE = DATA_DIR / "logs.txt"
MISC_FILE = DATA_DIR / "misc.txt"
TOKENS_TXT_FILE = DATA_DIR / "tokens.txt"
TOKENS_FILE = DATA_DIR / "tokens.npy"

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

    def count_tokens(self) -> int:
        """
        Counts the number of tokens in the tokenized data file
        
        :return: number of tokens in the tokenized data file
        :rtype: int
        """
        count = 0
        with open(TOKENS_TXT_FILE, "r", encoding="utf-8") as file:
            for line in file:
                tokens = line.split()
                count += len(tokens)

        return count

    def read_tokens(self) -> npt.NDArray:
        """
        reads tokens and returns them as a NumPy `array` of type `uint16`. Returns `None` if there was some error
        
        :return: tokens in the `TOKENS_FILE`
        :rtype: list[int]
        """
        try:
            return np.load(TOKENS_FILE)
        except OSError:
            return None


if __name__ == "__main__":
    pass

