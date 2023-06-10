from transformer_lens import HookedTransformer

from .diff import ModelDiff

def compare(dataset, *models):
    # TODO: Ensure models have the same tokenizer
    # TODO: Diff class should depend on whether models have the same architecture
    # TODO: Do we assume models are HookedTransformers?
    return ModelDiff(dataset, *models)