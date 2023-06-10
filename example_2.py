# %%
import urllib.request                                                           
from pathlib import Path
import json
from functools import partial

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

from modiff import compare

t.set_grad_enabled(False)
device = "cpu"  # t.device("cuda" if t.cuda.is_available() else "cpu")

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
    
    def add_perma_hooks_to_mask_pad_tokens(model: HookedTransformer, pad_token: int) -> HookedTransformer:
        import einops
        
        # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
        def cache_padding_tokens_mask(tokens, hook) -> None:
            hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == pad_token, "b sK -> b 1 1 sK")

        # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
        def apply_padding_tokens_mask(
            attn_scores,
            hook,
        ) -> None:
            attn_scores.masked_fill_(model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5)
            if hook.layer() == model.cfg.n_layers - 1:
                del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

        # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
        for name, hook in model.hook_dict.items():
            if name == "hook_tokens":
                hook.add_perma_hook(cache_padding_tokens_mask)
            elif name.endswith("attn_scores"):
                hook.add_perma_hook(apply_padding_tokens_mask)
    
    model.reset_hooks(including_permanent=True)
    add_perma_hooks_to_mask_pad_tokens(model, 1)
    return model

def tokenize(brackets, max_len):
    assert len(brackets) <= max_len, f"Too long text: {brackets}"
    assert set(list(brackets)).issubset(set(['(', ')'])), f"Tekst should contain only (), got {brackets}"
    
    out = [0]
    for ix, b in enumerate(brackets):
        out.append(3 + int(b == ')'))
    out.append(2)
    out += [1 for _ in range(max_len + 2 - len(out))]
        
    return t.tensor(out)
    

def get_dataset(subset_len=None):
    fname = "brackets_data.json"
    url = "https://drive.google.com/uc?export=download&id=1_05v9oAYjXyeaeSZv4KTzktYVpuK6xxH"
        
    if not Path(fname).exists():
        urllib.request.urlretrieve(url, fname)
    
    with open(fname, "r") as f:
        data = json.load(f)

    if subset_len is not None:
        data = data[:subset_len]
        
    ok, bad_cnt, bad_elevation = [], [], []
    for (brackets, is_ok) in data:
        if is_ok:
            ok.append(brackets)
        else:
            l_brackets = list(brackets)
            if l_brackets.count('(') != len(l_brackets) / 2:
                bad_cnt.append(brackets)
            else:
                bad_elevation.append(brackets)
                
    
    ok = t.stack([tokenize(x, 40) for x in ok])
    bad_cnt = t.stack([tokenize(x, 40) for x in bad_cnt])
    bad_elevation = t.stack([tokenize(x, 40) for x in bad_elevation])
       
    return [ok, bad_cnt, bad_elevation]
    
# %%
dataset = get_dataset(1000)

# %%
def ablate_head(v, hook, head_ix):
    v[:, :, head_ix, :] = 0.0

model_0 = get_model()
model_1 = get_model()
model_2 = get_model()

model_1.hook_dict[utils.get_act_name("v", 2)].add_perma_hook(partial(ablate_head, head_ix=0))
model_2.hook_dict[utils.get_act_name("v", 2)].add_perma_hook(partial(ablate_head, head_ix=1))

diff = compare(dataset, model_0, model_1, model_2)
diff.plot_pos_token_prob(0, 0).show()

# %%
