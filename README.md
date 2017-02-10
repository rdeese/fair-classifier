A Fair Logit Binary Classifier for scikit-learn
===============================================

`fair_logit_estimator.py` is an implementation of the fair classifier described
in [Learning Fair Classifiers (Zafar et al., 2016)](http://arxiv.org/abs/1507.05259v3).

## Usage

To use the classifier, make sure you have numpy, scipy, and scikit-learn installed. Then simply import `fair_logit_estimator.py` into your project.

Only binary dependent variables in `[-1, 1]` format are supported. Specify some number of sensitive attributes in your training data by passing a list of column indices. See code comments for more detailed documentation.

## Benchmarks

The `fair-classification` folder contains a lightly modified version of [mbilalzafar/fair-classification](https://github.com/mbilalzafar/fair-classification). Their testing code has been rewritten to benchmark the two implementations, proving that they produce nearly-identical output for the sample data.

To run the benchmarks:

```bash
cd fair-classification/synthetic_data_demo
python decision_boundary_demo.py
```
