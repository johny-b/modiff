from functools import cached_property


from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import plotly.express as px

from torch import Tensor
import torch as t

class ModelDiff:
    def __init__(self, dataset, *models):
        self.dataset = self._parse_raw_dataset(dataset)
        self.models = models
        
    @cached_property
    def logits(self) -> Float[Tensor, "models problems problem_examples seq_len d_voc"]:
        # TODO: maybe asyncio if models are on different devices?
        logits = []
        for i, model in enumerate(self.models):
            #   TODO: is this correct? Or maybe we should expect devices to be dealt with in the extenal code?
            device = next(model.parameters()).device
            dataset = self.dataset.to(device)
            
            # TODO: loop here doesn't make sense
            model_logits = []
            for problem in dataset:
                model_logits.append(model(problem))
            logits.append(t.stack(model_logits))
        return t.stack(logits)
    
    @cached_property
    def log_prob(self) -> Float[Tensor, "models problems seq_len"]:
        #   TODO: remove loop/loops
        all_log_probs = []
        for model_logits in self.logits:
            model_log_probs = []
            for (problem_logits, dataset) in zip(model_logits, self.dataset): 
                log_probs = problem_logits.log_softmax(dim=-1)
                log_probs = log_probs[:,:-1]  # remove last
                index = dataset[:, 1:].unsqueeze(-1)
                log_probs_for_tokens = log_probs.gather(dim=-1, index=index).squeeze(-1)
                model_log_probs.append(log_probs_for_tokens.mean(dim=0))
            all_log_probs.append(t.stack(model_log_probs))
        return t.stack(all_log_probs)
    
    @cached_property
    def log_prob_diff(self) -> Float[Tensor, "problems seq_len"]:
        if len(self.models) != 2:
            raise NotImplementedError("log_prob_diff is implemented only for 2-model scenario")
        return self.log_prob[0] - self.log_prob[1]
    
    def plot_log_prob_diff(self):
        data = self.log_prob_diff.cpu().numpy()              
        title = f"Per token log-prob on correct token, for sequence of length",
        labels = {"index": "Sequence position", "value": "Loss"}                  
        
        fig = px.line(data.T)  #, title=title, labels=labels)                  
        fig.update_layout(showlegend=False)                    
        # fig.add_vrect(x0=0, x1=seq_len-.5, fillcolor="red", opacity=0.2, line_width=0)
        # fig.add_vrect(x0=seq_len-.5, x1=2*seq_len-1, fillcolor="green", opacity=0.2, line_width=0)
        fig.show()
        
    def _parse_raw_dataset(self, dataset: Int[Tensor, "..."]) -> Int[Tensor, "problems problem_examples seq_len"]: 
        if len(dataset.shape) == 1:
            #   Just a sequence of tokens
            dataset = dataset.unsqueeze(0).unsqueeze(0)
        if len(dataset.shape) == 2:
            #   Batch of sequences of tokens
            dataset = dataset.unsqueeze(0)
        
        assert len(dataset.shape) == 3, "Dataset should have at most 3 dimensions"
        return dataset