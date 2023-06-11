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
    first_column = (t.ones(batch_size, 1) * tokenizer.bos_token_id).long()
    seq = t.randint(0, tokenizer.vocab_size, (batch_size, seq_len // 2), dtype=t.int64)
    return t.cat([first_column, seq, seq], dim=-1)

def generate_random_tokens(tokenizer, seq_len, batch_size):
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
# print(diff.max_attention_diff(5).shape)
diff.plot_max_attention_diff(5).show()
diff.plot_max_attention_diff(15).show()
# %%
