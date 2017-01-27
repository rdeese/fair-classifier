import os,sys
import numpy as np
from generate_synthetic_data import *
sys.path.insert(0, '../fair_classification/') # the code for fair classification is in this directory
sys.path.append('../..') # for FairLogitEstimator
import utils as ut
import loss_funcs as lf # loss funcs that can be optimized subject to various constraints
from fair_logit_estimator import FairLogitEstimator
import time



def test_synthetic_data():
    
    """ Generate the synthetic data """
    X, y, x_control = generate_synthetic_data(plot_data=False) # set plot_data to False to skip the data plot
    ut.compute_p_rule(x_control["s1"], y) # compute the p-rule in the original data


    """ Split the data into train and test """
    X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
    train_fold_size = 0.7
    x_train, y_train, x_control_train, x_test, y_test, x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)



    apply_fairness_constraints = None
    apply_accuracy_constraint = None
    sep_constraint = None

    loss_function = lf._logistic_loss
    sensitive_attrs = ["s1"]
    sensitive_attrs_to_cov_thresh = {}
    gamma = None

    def print_classifier_metrics(w, test_score, distances_boundary_test):
        all_class_labels_assigned_test = np.sign(distances_boundary_test)
        correlation_dict_test = ut.get_correlations(None, None, all_class_labels_assigned_test, x_control_test, sensitive_attrs)
        cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test, distances_boundary_test, x_control_test, sensitive_attrs)
        covariance, p_rule = ut.print_classifier_fairness_stats([test_score], [correlation_dict_test], [cov_dict_test], sensitive_attrs[0])    
        return w, covariance, p_rule, test_score

    def train_test_and_compare():
        print "= Original ="
        start = time.clock()
        _, o_cov, o_p_rule, o_test_score = train_test_classifier()
        end = time.clock()
        print("Finished in", end-start)
        print "= Sklearn ="
        start = time.clock()
        _, f_cov, f_p_rule, f_test_score = train_test_fair_logit()
        end = time.clock()
        print("Finished in", end-start)
        assert np.isclose(o_cov, f_cov, atol=1e-2)
        assert np.isclose(o_p_rule, f_p_rule, atol=1e-1)
        assert np.isclose(o_test_score, f_test_score, atol=1e-2)


    def train_test_classifier():
        w = ut.train_model(x_train, y_train, x_control_train, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy(w, x_train, y_train, x_test, y_test, None, None)
        distances_boundary_test = (np.dot(x_test, w)).tolist()
        return print_classifier_metrics(w, test_score, distances_boundary_test)

    def train_test_fair_logit():
        constraint = "none"
        sensitive_col_idx = 0
        if apply_fairness_constraints == 1: constraint = "fairness" 
        if apply_accuracy_constraint == 1: constraint = "accuracy" 
        x_train_copy = np.insert(x_train, sensitive_col_idx,
                                 values=x_control_train['s1'], axis=1)

        covariance_tolerance = sensitive_attrs_to_cov_thresh['s1'] if 'sex' in sensitive_attrs_to_cov_thresh else 0
        fle = FairLogitEstimator(constraint=constraint)
        fle.fit(x_train_copy, y_train,
                sensitive_col_idx=sensitive_col_idx,
                covariance_tolerance=covariance_tolerance,
                accuracy_tolerance=gamma)

        # add dummy sensitive column to test x data
        x_test_dummy = np.insert(x_test, sensitive_col_idx, values=0, axis=1)

        y_train_predicted = fle.predict(x_train_copy)
        y_test_predicted = fle.predict(x_test_dummy)

        train_score, test_score, correct_answers_train, correct_answers_test = ut.check_accuracy_from_results(y_train, y_test, y_train_predicted, y_test_predicted)
        distances_boundary_test = fle.boundary_distances(x_test_dummy).tolist()
        return print_classifier_metrics(fle.w_, test_score, distances_boundary_test)

    def plot_boundaries(w1, w2, p1, p2, acc1, acc2, fname):

        num_to_draw = 200 # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = y[:num_to_draw]
        x_control_draw = x_control["s1"][:num_to_draw]

        X_s_0 = x_draw[x_control_draw == 0.0]
        X_s_1 = x_draw[x_control_draw == 1.0]
        y_s_0 = y_draw[x_control_draw == 0.0]
        y_s_1 = y_draw[x_control_draw == 1.0]
        plt.scatter(X_s_0[y_s_0==1.0][:, 1], X_s_0[y_s_0==1.0][:, 2], color='green', marker='x', s=30, linewidth=1.5)
        plt.scatter(X_s_0[y_s_0==-1.0][:, 1], X_s_0[y_s_0==-1.0][:, 2], color='red', marker='x', s=30, linewidth=1.5)
        plt.scatter(X_s_1[y_s_1==1.0][:, 1], X_s_1[y_s_1==1.0][:, 2], color='green', marker='o', facecolors='none', s=30)
        plt.scatter(X_s_1[y_s_1==-1.0][:, 1], X_s_1[y_s_1==-1.0][:, 2], color='red', marker='o', facecolors='none', s=30)


        x1,x2 = max(x_draw[:,1]), min(x_draw[:,1])
        y1,y2 = ut.get_line_coordinates(w1, x1, x2)
        plt.plot([x1,x2], [y1,y2], 'c-', linewidth=3, label = "Acc=%0.2f; p%% rule=%0.0f%% - Original"%(acc1, p1))
        y1,y2 = ut.get_line_coordinates(w2, x1, x2)
        plt.plot([x1,x2], [y1,y2], 'b--', linewidth=3, label = "Acc=%0.2f; p%% rule=%0.0f%% - Constrained"%(acc2, p2))



        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc=2, fontsize=15)
        plt.xlim((-15,10))
        plt.ylim((-10,15))
        plt.savefig(fname)
        plt.show()


    """ Classify the data while optimizing for accuracy """
    print
    print "== Unconstrained (original) classifier =="
    # all constraint flags are set to 0 since we want to train an unconstrained (original) classifier
    apply_fairness_constraints = 0
    apply_accuracy_constraint = 0
    sep_constraint = 0
    train_test_and_compare()
    
    """ Now classify such that we optimize for accuracy while achieving perfect fairness """
    apply_fairness_constraints = 1 # set this flag to one since we want to optimize accuracy subject to fairness constraints
    apply_accuracy_constraint = 0
    sep_constraint = 0
    sensitive_attrs_to_cov_thresh = {"s1":0}
    print
    print "== Classifier with fairness constraint =="
    train_test_and_compare()
    # plot_boundaries(w_uncons, w_f_cons, p_uncons, p_f_cons, acc_uncons, acc_f_cons, "img/f_cons.png")


    # """ Classify such that we optimize for fairness subject to a certain loss in accuracy """
    # apply_fairness_constraints = 0 # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
    # apply_accuracy_constraint = 1 # now, we want to optimize fairness subject to accuracy constraints
    # sep_constraint = 0
    # gamma = 0.5 # gamma controls how much loss in accuracy we are willing to incur to achieve fairness -- increase gamme to allow more loss in accuracy
    # print "== Classifier with accuracy constraint =="
    # train_test_and_compare()
    # # plot_boundaries(w_uncons, w_a_cons, p_uncons, p_a_cons, acc_uncons, acc_a_cons, "img/a_cons.png")

    #  """ 
    #  Classify such that we optimize for fairness subject to a certain loss in accuracy 
    #  In addition, make sure that no points classified as positive by the unconstrained (original) classifier are misclassified.

    #  """
    #  apply_fairness_constraints = 0 # flag for fairness constraint is set back to0 since we want to apply the accuracy constraint now
    #  apply_accuracy_constraint = 1 # now, we want to optimize accuracy subject to fairness constraints
    #  sep_constraint = 1 # set the separate constraint flag to one, since in addition to accuracy constrains, we also want no misclassifications for certain points (details in demo README.md)
    #  gamma = 2000.0
    #  print "== Classifier with accuracy constraint (no +ve misclassification) =="
    #  train_test_and_compare()
    #  # w_a_cons_fine, p_a_cons_fine, acc_a_cons_fine  = train_test_classifier()
    #  # plot_boundaries(w_uncons, w_a_cons_fine, p_uncons, p_a_cons_fine, acc_uncons, acc_a_cons_fine, "img/a_cons_fine.png")

    return


def main():
    test_synthetic_data()


if __name__ == '__main__':
    main()
