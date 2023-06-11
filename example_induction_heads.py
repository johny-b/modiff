# %%
import torch as t
from transformer_lens import HookedTransformer

from modiff import compare

t.set_grad_enabled(False)
device = "cpu"  # t.device("cuda" if t.cuda.is_available() else "cpu")

# # DEV MODE: autoreload
# from IPython import get_ipython
# ipython = get_ipython()
# ipython.run_line_magic("load_ext", "autoreload")
# ipython.run_line_magic("autoreload", "2")

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
model_1l = HookedTransformer.from_pretrained("attn-only-1l").to(device)
model_2l = HookedTransformer.from_pretrained("attn-only-2l").to(device)

dataset = [
    generate_repeated_tokens(model_1l.tokenizer, seq_len=20, batch_size=10).to(device),
    generate_random_tokens(model_1l.tokenizer, seq_len=20, batch_size=10).to(device),
]

# %%
diff = compare(dataset, model_1l, model_2l)
diff.plot_correct_token_log_prob_diff().show()

print(f"""
OBSERVATIONS
*   Plot shows the average difference between log prob of the correct token 
    according to both models. 
*   Both models "predict" random data equally well (red line)
*   Second model is much better at predicting the repeated sequence 
    (second part of the blue line).
    
Example random sequence: {dataset[0][0]} - same tokens on positions n and n+10 for n>0.
""")


# %%
