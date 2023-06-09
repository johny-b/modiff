# %%

import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from IPython.display import display

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utils import to_numpy

from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import plotly.express as px

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%

model_1l = HookedTransformer.from_pretrained("attn-only-1l")
model_2l = HookedTransformer.from_pretrained("attn-only-2l")

# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    '''
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    '''
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    # SOLUTION
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens

def run_and_cache_model_repeated_tokens(model: HookedTransformer, seq_len: int, batch: int = 1) -> Tuple[t.Tensor, t.Tensor]:
    '''
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    '''
    # SOLUTION
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)
    rep_logits = model(rep_tokens)
    return rep_tokens, rep_logits

def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], 
    tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

def plot_loss_difference(log_probs, seq_len):                          
    fig = px.line(                                                              
        to_numpy(log_probs),                           
        title=f"Per token log-prob on correct token, for sequence of length {seq_len}*2 (repeated twice)",
        labels={"index": "Sequence position", "value": "Loss"}                  
    ).update_layout(showlegend=False, hovermode="x unified")                    
    fig.add_vrect(x0=0, x1=seq_len-.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=seq_len-.5, x1=2*seq_len-1, fillcolor="green", opacity=0.2, line_width=0)
    fig.show() 

seq_len = 50
batch = 50
log_probs = []

for model in (model_1l, model_2l):
    (rep_tokens, rep_logits) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    model.reset_hooks()
    batch_log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()
    log_probs.append(batch_log_probs.mean(dim=0))

for x in log_probs:
    plot_loss_difference(x, seq_len)
# %%

diff = log_probs[0] - log_probs[1]
plot_loss_difference(diff, seq_len)
# %%
