# %%
import torch as t
from transformer_lens import HookedTransformer

from modiff import compare

t.set_grad_enabled(False)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# DEV MODE: autoreload
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# %%
def generate_repeated_tokens(tokenizer, seq_len, batch_size):
    first_column = (t.ones(batch_size, 1) * tokenizer.bos_token_id).long()
    seq = t.randint(0, tokenizer.vocab_size, (batch_size, seq_len // 2), dtype=t.int64)
    return t.cat([first_column, seq, seq], dim=-1)

def generate_random_tokens(tokenizer, seq_len, batch_size):
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
diff.plot_log_prob_diff().show()

# %%
