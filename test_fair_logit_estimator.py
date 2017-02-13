import pytest
import numpy as np
from fair_logit_estimator import _get_fairness_constraints, _get_fairness_constraint

class TestFairnessConstraint:
    def test_perfect_correlation(self):
        sensitive_x = np.array([0,1,0,0,1,0,1,0,0,1])
        unsensitive_x = np.array(list(map(lambda x: [x, 0], sensitive_x)))
        w = np.array([1,0])
        constraint = _get_fairness_constraint(unsensitive_x, sensitive_x, 0)
        constraint_fn = constraint['fun']
        assert constraint_fn(w, unsensitive_x, sensitive_x, 0) == -1

    def test_perfect_anticorrelation(self):
        sensitive_x = np.array([0,1,0,0,1,0,1,0,0,1])
        unsensitive_x = np.array(list(map(lambda x: [x, 0], sensitive_x)))
        w = np.array([-1,0])
        constraint = _get_fairness_constraint(unsensitive_x, sensitive_x, 0)
        constraint_fn = constraint['fun']
        assert constraint_fn(w, unsensitive_x, sensitive_x, 0) == -1

    def test_zero_correlation(self):
        NUM_SAMPLES = 100000
        sensitive_x = np.array([np.random.random() for _ in range(NUM_SAMPLES)])
        unsensitive_x = np.array([[np.random.random(), np.random.random()] for _ in range(NUM_SAMPLES)])
        w = np.array([1,1])
        constraint = _get_fairness_constraint(unsensitive_x, sensitive_x, 0)
        constraint_fn = constraint['fun']
        assert np.isclose(constraint_fn(w, unsensitive_x, sensitive_x, 0), 0, atol=0.01)
