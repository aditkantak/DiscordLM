# DiscordLM
training a small-scale character-tokenized transformer language model on discord chat logs. Transformer implementation based on [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762) (original transformer paper), [_Improving Language Understanding by Generative Pre-Training_](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (original GPT paper), and [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT).

### Steps of the process
1. clean up data
2. design data pipeline
3. design model
4. train/eval <----- here currently

### Current Status
- Working out hyperparams and training on [Lambda Labs](https://www.lambda.ai) A10 instances (find most updated scripts in `cloud/` folder)
- Also working on optimizing selection from final probabilities

## Update 1/24/26
#### Best achieved val loss: 1.083104237977642
- Model is finally coherent(ish)!!!! scroll down for some funny quotes.
- Surprisingly doesn't generate too much personal information, I'm going to test it more and if it's not bad, I might make a little playground for the website

### Training data not included for privacy. Information on how to get your own data is in `guide.md`

### Next Steps (more concrete):
- replace relu with gelu
- better learning rate scheduler
- more optimal weight initialization
- try changing up vocab to make more expressive

### Things to explore (broader next steps in no order):
- maybe some basic RL? (not sure how significant if at all gains would be)
- data without line breaks, dataset as series of sequences of varying length
- sinusoidal vs learned positional encodings
- fixed vs learned attention scaling factor
- different embedding dimensions/query dimensions across transformer blocks

### Funny Quotes
> research is so funny <br>
> but the problem is that its a good thing <br>
> i was thinking of getting a different problem <br>
> i mean its a little slow so i can get over it

\- AditGPT 2026 (Prompted with "research")