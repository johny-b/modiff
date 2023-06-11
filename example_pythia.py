# %%
import torch as t
from transformer_lens import HookedTransformer

from modiff import compare

t.set_grad_enabled(False)
device = "cpu"  # t.device("cuda" if t.cuda.is_available() else "cpu")

# DEV MODE: autoreload
# from IPython import get_ipython
# ipython = get_ipython()
# ipython.run_line_magic("load_ext", "autoreload")
# ipython.run_line_magic("autoreload", "2")

# %%
pythia_early = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", checkpoint_value=512).to(device)
pythia_late = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", checkpoint_value=1000).to(device)

# %%
def generate_repeated_tokens(tokenizer, seq_len, batch_size):
    """Return a repeated sequence of random tokens, total length (1 + seq_len) (1 for BOS)."""
    first_column = (t.ones(batch_size, 1) * tokenizer.bos_token_id).long()
    seq = t.randint(0, tokenizer.vocab_size, (batch_size, seq_len // 2), dtype=t.int64)
    return t.cat([first_column, seq, seq], dim=-1)

def generate_random_tokens(tokenizer, seq_len, batch_size):
    """Return a sequence of random tokens, total length (1 + seq_len) (1 for BOS)."""
    first_column = (t.ones(batch_size, 1) * tokenizer.bos_token_id).long()
    seq = t.randint(0, tokenizer.vocab_size, (batch_size, seq_len), dtype=t.int64)
    return t.cat([first_column, seq], dim=-1)

# %%
dataset = [
    generate_repeated_tokens(pythia_late.tokenizer, seq_len=20, batch_size=100).to(device),
    generate_random_tokens(pythia_late.tokenizer, seq_len=20, batch_size=50).to(device),
]
# assert False
# %%
diff = compare(dataset, pythia_early, pythia_late)
plot = diff.plot_correct_token_log_prob_diff().show()

# %%
diff.plot_max_attention_diff(15).show()

print(f"""OBSERVATIONS
*   We have two early pythia models, after 512 and 1000 training steps
*   First we do the same analysis as in `example_induction_heads.py`. We notice that:
    *   Younger pythia is "better" at "predicting" random sequences (red line). This is not surprising,
        the better the model the more unlikely should be a random sequence.
    *   Older pythia is clearly much better at predicting the repeated sequence.
    *   We conclude that induction heads appeared somewehere between 512 and 1000 training step.
*   Later we plot the difference in average maximum attention per head, on position 15 (i.e. in the repeated part of the sequence).
    *   Just to clarify how this is calculated:
        *   We get all attention patterns
        *   We look at the attention of token 15 (passed as a parameter)
        *   We take maximum value of the attention. High value is "I look in a particular place", low value is "I look everywhere".
        *   We average this maximum value per batch
        *   We calculate this for all heads and both models and subtract
    *   We observe that:
        *   For the repeated sequence (blue line):
            *   There's no significant difference in behaviour of heads in layers (0,1,2) except 2.7
            *   The biggest difference between models is for heads 2.7 3.0, 3.1, 3.6, 4.6, 4.7
            *   In all the cases heads in the later model are more certain what token they
                should pay attention to.
        *   For the random sequence (red line): 
            *   Head 2.7 behaves similarly as in the repeated sequence (blue line)
            *   Heads 3.0, 3.1, 3.6 are totally different, heads 4.6 and 4.7 prety much too.
        *   To conclude:
            *   Head 3.0 behavior differs between models and also between random/repeated sequences.
                Therefore it is a (last?) part of the induction head circuit.
            *   Same for 3.1, 3.6, 4.6 and 4.7
            *   Head 2.7 changed a lot between earlier/later model, but doesn't matter for the
                induction circuit (I'm not sure about this though)""")
# %%
