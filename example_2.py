# %%
import urllib.request                                                           
from pathlib import Path
import json

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

from modiff import compare

t.set_grad_enabled(False)
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# DEV MODE: autoreload
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# %%
def get_model():
    cfg = HookedTransformerConfig(
        n_ctx=42,
        d_model=56,
        d_head=28,
        n_heads=2,
        d_mlp=56,
        n_layers=3,
        attention_dir="bidirectional",
        act_fn="relu",
        d_vocab=5,
        d_vocab_out=2,
        use_attn_result=True, 
        device=device,
        use_hook_tokens=True
    )

    model = HookedTransformer(cfg).eval()
    
    fname = "brackets_model_state_dict.pt"
    url = "https://drive.google.com/uc?export=download&id=1ahvkvczt6UCNqM21N_vADimdo5w1SPD6"
    if not Path(fname).exists():
        urllib.request.urlretrieve(url, fname) 

    state_dict = t.load("brackets_model_state_dict.pt")
    model.load_state_dict(state_dict)
    return model

def tokenize(brackets, max_len):
    assert len(brackets) <= max_len, f"Too long text: {brackets}"
    assert set(list(brackets)).issubset(set(['(', ')'])), f"Tekst should contain only (), got {brackets}"
    
    out = [0] + [1] * max_len + [2]
    for ix, b in enumerate(brackets):
        out[ix + 1] = 3 + int(b == '(')
        
    return out
    

def get_dataset(tokenizer):
    fname = "brackets_data.json"
    url = "https://drive.google.com/uc?export=download&id=1_05v9oAYjXyeaeSZv4KTzktYVpuK6xxH"
        
    if not Path(fname).exists():
        urllib.request.urlretrieve(url, fname)
    
    with open(fname, "r") as f:
        data = json.load(f)
        
    ok, bad_cnt, bad_elevation = [], [], []
    for (brackets, is_ok) in data[:7]:
        if is_ok:
            ok.append(brackets)
        else:
            l_brackets = list(brackets)
            if l_brackets.count('(') != len(l_brackets) / 2:
                bad_cnt.append(brackets)
            else:
                bad_elevation.append(brackets)
                
    
    ok = t.stack([t.tensor(tokenize(x, 40)) for x in ok])
    bad_cnt = t.stack([t.tensor(tokenize(x, 40)) for x in bad_cnt])
    bad_elevation = t.stack([t.tensor(tokenize(x, 40)) for x in bad_elevation])
       
    return t.stack((ok, bad_cnt, bad_elevation))
    
# %%
dataset = get_dataset(1)

# %%
model = get_model()
print(model.tokenizer)
# %%
