# DiscordLM Guide
Here's a little guide on how to get the data ready for the model training, and a little explanation of what's happening under the hood.
## How to get and format your discord data correctly
1. Follow the instructions at [this link](https://support.discord.com/hc/en-us/articles/360004027692-Requesting-a-Copy-of-your-Data) to securely request your message data from Discord. It can take up to 30 business days for them to process your request and send you the link to download your data.
2. When you get the email with the link to download your data, save the `.zip` file into a new folder in this repo called `raw_discord/`
3. Extract the files into the `raw_discord/` folder. They should separate into a `Messages/` directory and a `Servers/` directory.
4. Run `clean_data.py`.
5. The data is cleaned and will be saved in the `data/` directory.

## `data/` after cleaning
Messages that are removed in the cleaning process:
1. Messages that only consist of mention(s)
2. Messages that contain URLs
3. Messages with code blocks
4. Messages with line breaks
5. Messages that only consist of emoji(s)

### Contents of `data/`
- `clean_discord.txt` - all the cleaned messages from your raw data.
- `vocab.txt` - a complete vocabulary of all characters occurring in the cleaned data, accompanied by their counts, sorted by most common to least common. Use for reference when setting vocabulary size for the model.
- `logs.txt` (optional) - all raw discord messages ever sent in a more readable format (not used in model training).
