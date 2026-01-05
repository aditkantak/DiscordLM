# DiscordLM
training a little transformer language model on discord chat logs. Transformer implementation based on _Attention Is All You Need_ paper and Karpathy's NanoGPT.

Steps of the process
1. clean up data
2. design model <----- here currently
3. train/eval

Data hidden for obvious privacy reasons. Will keep process documented.

Things to explore:
- data without line breaks
- dataset as series of sequences of varying length
- fixed vs learned positional encodings
- fixed vs learned attention scaling factor
- different embedding dimensions/query dimensions across transformer blocks