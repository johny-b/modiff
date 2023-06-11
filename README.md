# modiff - Model Difference analysis

Compare two or more models. Get some useful data / plots. Built on top of TransformerLens.

This is a hackaton-mode early alpha.

## Purpose

### Idea

There is a set of [open problems in mech interp](https://www.alignmentforum.org/s/yivyHaCAmMJ3CqSyj/p/btasQF7wiCYPsr5qw) that can be summarized as "build a model-comparing tool and test it" (6.49 - 6.58). This is a POC of such a tool. 

### Purpose

Main two workflows:

* Exploration. You have a model and you want to compare its behaviour to behaviour of some model.
* Hypothesis testing. You want to test some hypothesis and you can express it in terms of difference between two models.


## Examples

* `example_induction_heads.py` - Compare `attn-only-1l` and `attn-only-2l` models. Guess which one has induction heads!
* `example_pythia.py` - Compare early `pythia` models, after 512 and 1000 steps of learning:
    * Notice that induction heads appeared in the later one
    * Find the suspected heads
* `example_brackets.py` - Take two copies of the bracket classifier model. Ablate a certain head in one of them. Compare their performance.  

More details inside examples.