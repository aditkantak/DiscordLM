# DiscordLM
training a little transformer language model on discord chat logs. Transformer implementation based on [_Attention Is All You Need_](https://arxiv.org/abs/1706.03762) (original transformer paper), [_Improving Language Understanding by Generative Pre-Training_](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (original GPT paper), and [Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT).

### Steps of the process
1. clean up data
2. design data pipeline
3. design model
4. train/eval <----- here currently

### Current Status
- Working out hyperparams and training on [Lambda Labs](https://www.lambda.ai) A10 instances (find most updated scripts in `cloud/` folder)
- Also working on optimizing selection from final probabilities; implementing temperature
- Kind of side item, but wanna try benchmarking my new macbooks neural engine, have to look into using mps accelerator

## Update 1/24/26
#### Best achieved val loss: 1.083104237977642
- Model is finally coherent(ish)!!!! scroll down for example output.
- Surprisingly doesn't generate too much personal information, I'm going to test it more and if it's not bad, I might make a little playground for the website
- Trained for 10 epochs, but logs are still stuck on stupid Arizona server storage, I'll have to wait until night to get it out (need to switch gpu providers or figure out some better pipeline) to see the loss curve.

### Training data not included for privacy. Information on how to get your own data is in `guide.md`

### Things to explore:
- data without line breaks
- dataset as series of sequences of varying length
- sinusoidal vs learned positional encodings
- fixed vs learned attention scaling factor
- different embedding dimensions/query dimensions across transformer blocks

### Example Output
tmr by then <br>
huh these bozos aint no to me<br>
who tf are we saw that<br>
u gotta listen to<br>
its my message bruh<br>
im not loading glory math today<br>
lmaoaoaoo he wants to listen to<br>
make him something else and would u have to<br>
keep the 30 mode<br>
mr band dad comes back on i got<br>
i will not add fit<br>
nah do we have to do it together for the stuff<br>
or integrated ggive them<br>
if ur headset then u buy another card<br>
with the adp respect that out and make you<br>
loose loser<br>
the old og school<br>
wait what are the only non purpose of the<br>
other switches i just do in the most bajtl those<br>
bozos<br>
mad cute ngl<br>
so i wouldnt leave my house<br>
u can try to have a snowflake omg..xxym thing<br>