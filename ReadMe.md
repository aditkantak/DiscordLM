# DiscordLM
training a little transformer language model on discord chat logs. Transformer implementation based on [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762) (original transformer paper), [_Improving Language Understanding by Generative Pre-Training_](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (original GPT paper), and [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT).

Steps of the process
1. clean up data
2. design data pipeline
3. design model
4. train/eval <----- here currently

Data hidden for obvious privacy reasons. Will keep process documented.

Things to explore:
- data without line breaks
- dataset as series of sequences of varying length
- sinusoidal vs learned positional encodings
- fixed vs learned attention scaling factor
- different embedding dimensions/query dimensions across transformer blocks