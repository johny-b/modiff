from functools import cached_property


from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import plotly.express as px
from plotly.graph_objects import Figure
from torch import Tensor
import torch as t


class ModelDiff:
    def __init__(self, dataset, *models):
        self.dataset = self._parse_raw_dataset(dataset)
        self.models = models
        
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
    
    def pos_token_prob(self, pos_ix: int, out_x: int) -> Float[Tensor, "problems models"]:
        out = []
        for logits in self.logits:
            logits = logits[:, :, pos_ix, :]
            probs = logits.log_softmax(dim=-1)
            probs = probs[:, :, out_x].mean(dim=1)
            out.append(probs)
        return t.stack(out)
        
    
    @cached_property
    def log_prob(self) -> Float[Tensor, "problems models seq_len"]:
        #   TODO: remove loop/loops
        all_log_probs = []
        for problem_logits, dataset in zip(self.logits, self.dataset):

            model_log_probs = []
            for model_logits in problem_logits:
                log_probs = model_logits.log_softmax(dim=-1)
                log_probs = log_probs[:,:-1]  # remove last
                index = dataset[:, 1:].unsqueeze(-1)
                log_probs_for_tokens = log_probs.gather(dim=-1, index=index).squeeze(-1)
                model_log_probs.append(log_probs_for_tokens.mean(0))
            all_log_probs.append(t.stack(model_log_probs))
        return t.stack(all_log_probs)
    
    @cached_property
    def log_prob_diff(self) -> Float[Tensor, "problems seq_len"]:
        if len(self.models) != 2:
            raise NotImplementedError("log_prob_diff is implemented only for 2-model scenario")
        return self.log_prob[:,0] - self.log_prob[:,1]

    def plot_log_prob_diff(self) -> Figure:
        data = self.log_prob_diff.cpu().numpy()
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
            title=f"Log prob value for token {out_ix} on position {pos_ix}",
            legend_title="Problem id",
            xaxis_title="Model ix",
            yaxis_title="Log prob",
            title_x=0.5,
        )                    
        return fig
        
    def _parse_raw_dataset(self, dataset) -> List[Int[Tensor, "batch seq_len"]]:
        # FIXME
        return dataset
        
        if len(dataset.shape) == 1:
            #   Just a sequence of tokens
            dataset = dataset.unsqueeze(0).unsqueeze(0)
        if len(dataset.shape) == 2:
            #   Batch of sequences of tokens
            dataset = dataset.unsqueeze(0)
        
        assert len(dataset.shape) == 3, "Dataset should have at most 3 dimensions"
        return dataset