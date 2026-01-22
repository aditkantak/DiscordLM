# DiscordLM
training a little transformer language model on discord chat logs. Transformer implementation based on [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762) (original transformer paper), [_Improving Language Understanding by Generative Pre-Training_](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (original GPT paper), and [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT).

### Current Status
- Working out hyperparams and training on [Lambda Labs](https://www.lambda.ai) A10 instances (find most updated scripts in `cloud/` folder)
- Best val loss ~1.3 (don't think it's converged)
- Also working on optimizing selection from final probabilities; implementing temperature
- Kind of side item, but wanna try benchmarking my new macbooks neural engine, have to look into using mps accelerator

### Steps of the process
1. clean up data
2. design data pipeline
3. design model
4. train/eval <----- here currently

Training data not included for privacy. Information on how to get your own data is in `guide.md`

Things to explore:
- data without line breaks
- dataset as series of sequences of varying length
- sinusoidal vs learned positional encodings
- fixed vs learned attention scaling factor
- different embedding dimensions/query dimensions across transformer blocks