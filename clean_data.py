import json
import os
import re
import emoji
from pathlib import Path

BASE_DIR = Path(".")
DIRECTORY_PATH = BASE_DIR / "raw_discord" / "Messages"
DATA_DIR = BASE_DIR / "data"

DATA_FILE = DATA_DIR / "clean_discord.txt"
VOCAB_FILE = DATA_DIR / "vocab.txt"
LOGS_FILE = DATA_DIR / "logs.txt"
MISC_FILE = DATA_DIR / "misc.txt"

# bad_words = set(["fuck", "shit", "bitch"])

unique_strings = set()

def parse_discord(input_dir):
    file_path = input_dir / "messages.json"

    with open(file_path, "r", encoding = "utf-8") as file:
        data = json.load(file)

        with open(DATA_FILE, "a", encoding = "utf-8") as out:
            for element in data: #of form {"ID": int, "Timestamp": datetime, "Contents": str, "Attachments": str}
                #messages_json_data.append({"message": element["Contents"], "Timestamp": element["Timestamp"]})
                text = clean_message(element["Contents"])
                if text and (text not in unique_strings):
                    unique_strings.add(text)
                    out.write(text + "\n")
                    



def clean_message(raw_text: str):
    url_regex = r'https?://\S+|www\.\S+' #to remove urls
    mention_regex = r'<@!?&?\d+>' #to remove discord mentions
    code_regex = r'```[\s\S]*```' #to remove code snippets
    multiline_regex = r'.*\n.*' #to remove multiline messages (usually unrelated)
    onlyemoji_regex = r'^(\\u[0-9a-fA-F]{4})+$'#to remove lines of only emojis
    customemote_regex = r'<a?:\w+:\d+>' #message reference is <#xxxxx>
    if (re.search(url_regex, raw_text) is not None)\
        or (re.search(mention_regex, raw_text) is not None)\
        or (re.search(code_regex, raw_text) is not None)\
        or (re.search(multiline_regex, raw_text) is not None)\
        or (re.search(onlyemoji_regex, raw_text) is not None)\
        or (re.search(customemote_regex, raw_text) is not None)\
        or (len(raw_text.split()) < 3):
        return ""
    for char in raw_text:
        if (not char.isascii() and not emoji.is_emoji(char)):
            return ""
    return raw_text.casefold()
    
def count_words():
    word_counts = {}
    with open(DATA_FILE, "r", encoding = "utf-8") as file:
        for line in file:
            for token in line: #tokens:
                if token in word_counts:
                    word_counts[token] += 1
                else:
                    word_counts[token] = 1
    
    with open(VOCAB_FILE, "a", encoding = "utf-8") as file:
        sorted_words = sorted(word_counts.keys(), key=(lambda key: word_counts[key]), reverse=True)
        for word in sorted_words:
            file.write(f"{word}: {word_counts[word]}\n")
        
def create_server_index():
    index_path = "./raw_discord/Messages/index.json"
    with open(index_path, "r", encoding = "utf-8") as file:
        server_names = json.load(file)

    return server_names

def create_logs():
    server_names = create_server_index()

    def server_name(file_path):
        with open(file_path, "r", encoding = "utf-8") as file:
            data = json.load(file)
            return server_names[data["id"]]

    for entry_name in os.listdir(DIRECTORY_PATH):
        #full_path = os.path.join(DIRECTORY_PATH, entry_name)
        full_path = DIRECTORY_PATH / entry_name
        
        if os.path.isdir(full_path):

            message_file = full_path / "messages.json"
            server_info_file = full_path / "channel.json"

            with open(message_file, "r", encoding = "utf-8") as file:
                data = json.load(file)
                server = server_name(server_info_file)
                with open(LOGS_FILE, "a", encoding = "utf-8") as out:
                    for element in data: #of form {"ID": int, "Timestamp": datetime, "Contents": str, "Attachments": str}
                        #messages_json_data.append({"message": element["Contents"], "Timestamp": element["Timestamp"]})
                        out.write(f"@{element["Timestamp"]} in {server} \"{element["Contents"]}\"\n")


if __name__ == "__main__":
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

    if os.path.exists(VOCAB_FILE):
        os.remove(VOCAB_FILE)

    if os.path.exists(MISC_FILE):
        os.remove(MISC_FILE)

    server_names = create_server_index()

    for entry_name in os.listdir(DIRECTORY_PATH):

        full_path = DIRECTORY_PATH / entry_name

        if os.path.isdir(full_path):
            parse_discord(full_path)

    count_words()