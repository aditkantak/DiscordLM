import json
import os
import re
import emoji

#discord
directory_path = "./raw/Messages/"
output_file = "./data/clean_discord.txt"
vocab_file = "./data/vocab.txt"
data_file = "./data/logs.txt"
misc_file = "./data/misc.txt"

bad_words = set(["fuck", "shit", "bitch"])

unique_strings = set()

def parse_discord(input_dir):
    file_path = os.path.join(input_dir, "messages.json")
    with open(file_path, "r") as file:
        data = json.load(file)
        with open(output_file, "a") as out:
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
    with open(output_file, "r") as file:
        for line in file:
            #tokens = line.split()
            for token in line: #tokens:
                if token in word_counts:
                    word_counts[token] += 1
                else:
                    word_counts[token] = 1
    
    with open(vocab_file, "a") as file:
        sorted_words = sorted(word_counts.keys(), key=(lambda key: word_counts[key]), reverse=True)
        for word in sorted_words:
            file.write(f"{word}: {word_counts[word]}\n")
        
def create_server_index():
    index_path = "./raw/raw_discord/Messages/index.json"
    with open(index_path, "r") as file:
        server_names = json.load(file)

    return server_names

def create_logs():
    server_names = create_server_index()

    def server_name(file_path):
                with open(file_path, "r") as file:
                    data = json.load(file)
                    return server_names[data["id"]]

    for entry_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry_name)
        
        if os.path.isdir(full_path):

            message_file = os.path.join(full_path, "messages.json")
            server_info_file = os.path.join(full_path, "channel.json")

            with open(message_file, "r") as file:
                data = json.load(file)
                server = server_name(server_info_file)
                with open(data_file, "a") as out:
                    for element in data: #of form {"ID": int, "Timestamp": datetime, "Contents": str, "Attachments": str}
                        #messages_json_data.append({"message": element["Contents"], "Timestamp": element["Timestamp"]})
                        out.write(f"@{element["Timestamp"]} in {server} \"{element["Contents"]}\"\n")


if __name__ == "__main__":
    i = 1
    num_files = 100000000000

    if os.path.exists(output_file):
        os.remove(output_file)

    if os.path.exists(vocab_file):
        os.remove(vocab_file)

    if os.path.exists(misc_file):
        os.remove(misc_file)

    server_names = create_server_index()

    for entry_name in os.listdir(directory_path):
        if (i > num_files):
            break
        i += 1

        full_path = os.path.join(directory_path, entry_name)
        if os.path.isdir(full_path):
            parse_discord(full_path)

    count_words()