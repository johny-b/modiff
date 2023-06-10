from functools import cached_property
from transformer_lens.utils import to_numpy

import plotly.express as px

import torch as t

class ModelDiff:
    def __init__(self, datatset, *models):
        self.dataset = datatset
        self.models = models
        
    @cached_property
    def logits(self):
        # TODO: maybe asyncio if models are on different devices?
        logits = []
        for i, model in enumerate(self.models):
            device = next(model.parameters()).device
            dataset = self.dataset.to(device)
            logits.append(model(dataset))
        return t.stack(logits)
    
    @cached_property
    def log_prob(self):
        all_log_probs = []
        for model_logits in self.logits: 
            log_probs = model_logits.log_softmax(dim=-1)
            log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=self.dataset[:, 1:].unsqueeze(-1)).squeeze(-1)
            all_log_probs.append(log_probs_for_tokens.mean(dim=0))
        return t.stack(all_log_probs)
    
    @cached_property
    def log_prob_diff(self):
        if len(self.models) != 2:
            raise NotImplementedError("log_prob_diff is implemented only for 2-model scenario")
        return self.log_prob[0] - self.log_prob[1]
    
    def plot_log_prob_diff(self):
        data = self.log_prob_diff.cpu().numpy()              
        title = f"Per token log-prob on correct token, for sequence of length",
        labels = {"index": "Sequence position", "value": "Loss"}                  
        
        fig = px.line(data)  #, title=title, labels=labels)                  
        fig.update_layout(showlegend=False)                    
        # fig.add_vrect(x0=0, x1=seq_len-.5, fillcolor="red", opacity=0.2, line_width=0)
        # fig.add_vrect(x0=seq_len-.5, x1=2*seq_len-1, fillcolor="green", opacity=0.2, line_width=0)
        fig.show() 