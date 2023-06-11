from functools import cached_property
from typing import List, Optional, Tuple

from jaxtyping import Int, Float
import plotly.express as px
from plotly.graph_objects import Figure
from torch import Tensor
import torch as t
from transformer_lens import ActivationCache

class ModelDiff:
    def __init__(self, dataset, *models):
        self.dataset = self._parse_raw_dataset(dataset)
        self.models = models
        
    #################################
    #   TRANSFORMERLENS STUFF
    @cached_property
    def activation_cache(self) -> List[List[ActivationCache]]:
        out = []
        for dataset in self.dataset:
            dataset_cache = []
            for model in self.models:
                _, cache = model.run_with_cache(dataset)
                dataset_cache.append(cache)
            out.append(dataset_cache)
        return out    

    def max_attention_diff(self, pos) -> Float[Tensor, "tasks n_heads"]:
        if len(self.models) != 2:
            raise NotImplementedError("max_attention_diff is implemented only for 2-model scenario")
        
        #   TODO: ensure models have the same architecture
        config = self.models[0].cfg
        n_layers, n_heads = config.n_layers, config.n_heads
        
        out_vals = []
        for dataset_caches in self.activation_cache:
            cache_1, cache_2 = dataset_caches
            vals = []
            for layer in range(n_layers):
                for head in range(n_heads):
                    pattern_1 = cache_1["pattern", layer]
                    pattern_2 = cache_2["pattern", layer]
                    max_attn_1 = pattern_1[:, head, pos].max(dim=-1).values.mean().item()
                    max_attn_2 = pattern_2[:, head, pos].max(dim=-1).values.mean().item()
                    vals.append(max_attn_1 - max_attn_2)
            out_vals.append(vals)
        return t.tensor(out_vals)

    #################################
    #   GENERAL PROPERTIES/FUNCTIONS
    @cached_property
    def logits(self) -> List[Float[Tensor, "models batch seq_len d_voc"]]:
        # TODO: maybe asyncio if models are on different devices?
        logits = []
        for problem in self.dataset:
            model_logits = []
            for i, model in enumerate(self.models):
                # TODO: Do we care about devices here?
                # TODO: Loop here doesn't make sense
                model_logits.append(model(problem))
            logits.append(t.stack(model_logits))
        return logits
    
    @cached_property
    def log_probs(self) -> List[Float[Tensor, "models batch seq_len d_voc"]]:
        log_probs = []
        for problem_logits, dataset in zip(self.logits, self.dataset):
            model_log_probs = []
            for model_logits in problem_logits:
                model_log_probs.append(model_logits.log_softmax(dim=-1))
            log_probs.append(t.stack(model_log_probs))
        return log_probs
        
    @cached_property
    def correct_token_log_prob(self) -> Float[Tensor, "problems models seq_len"]:
        cfg = self.models[0].cfg
        if cfg.d_vocab != cfg.d_vocab_out:
            raise RuntimeError("correct_token_log_prob requires d_vocab == d_vocab_out")

        all_log_probs = []
        for x, dataset in zip(self.log_probs, self.dataset):
            model_log_probs = []
            for log_probs in x:
                log_probs = log_probs[:,:-1]  # remove last
                index = dataset[:, 1:].unsqueeze(-1)
                log_probs_for_tokens = log_probs.gather(dim=-1, index=index).squeeze(-1)
                model_log_probs.append(log_probs_for_tokens.mean(0))
            all_log_probs.append(t.stack(model_log_probs))
        return t.stack(all_log_probs)
    
    @cached_property
    def correct_token_log_prob_diff(self) -> Float[Tensor, "problems seq_len"]:
        if len(self.models) != 2:
            raise NotImplementedError("log_prob_diff is implemented only for 2-model scenario")
        return self.correct_token_log_prob[:,0] - self.correct_token_log_prob[:,1]

    def pos_token_prob(self, pos_ix: int, out_x: int) -> Float[Tensor, "problems models"]:
        out = []
        for log_probs in self.log_probs:
            log_probs = log_probs[:, :, pos_ix, :]
            out.append(log_probs[:, :, out_x].mean(dim=1))
        return t.stack(out)

    #################################
    #   PLOTS
    def plot_correct_token_log_prob_diff(self) -> Figure:
        data = self.correct_token_log_prob_diff.cpu().numpy()
        fig = px.line(data.T)                
        fig.update_layout(
            title="Difference of log probs of the correct token between models",
            legend_title="Problem id",
            xaxis_title="Token position",
            yaxis_title="Correct token log prob diff",
            title_x=0.5,
        )                    
        return fig

    def plot_pos_token_prob(self, pos_ix: int, out_ix: int) -> Figure:
        data = self.pos_token_prob(pos_ix, out_ix).cpu().numpy()
        fig = px.bar(data.T, barmode="group")                
        fig.update_layout(
            title=f"Average log prob value for token {out_ix} on position {pos_ix}",
            legend_title="Problem id",
            xaxis_title="Model ix",
            yaxis_title="Log prob",
            title_x=0.5,
        )                    
        return fig
    
    def plot_max_attention_diff(self, pos) -> Figure:
        data = self.max_attention_diff(pos)
        fig = px.line(data.T)                
        fig.update_layout(
            title=f"Average log prob value for token",
            legend_title="Problem id",
            xaxis_title="Model ix",
            yaxis_title="Log prob",
            title_x=0.5,
        )                    
        return fig
    
    ##################################
    #   PRIVATE
    def _parse_raw_dataset(self, dataset) -> List[Int[Tensor, "batch seq_len"]]:
        # FIXME
        return dataset