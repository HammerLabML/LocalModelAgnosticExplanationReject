from random import Random
import sys
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from modelselection_conformal import ConformalRejectOptionGridSearchCV

from utils import *
from conformalprediction import ConformalPredictionClassifier, ConformalPredictionClassifierRejectOption, MyClassifierSklearnWrapper
from explanation import Explanation


def get_model(model_desc):
    if model_desc == "knn":
        return KNeighborsClassifier
    elif model_desc == "randomforest":
        return RandomForestClassifier
    elif model_desc == "gnb":
        return GaussianNB
    else:
        raise ValueError(f"Invalid value of 'model_desc' -- must be one of the following 'knn', 'dectree', 'gnb'; but not '{model_desc}'")

def get_model_parameters(model_desc):
    if model_desc == "knn":
        return knn_parameters
    elif model_desc == "randomforest":
        return random_forest_parameters
    elif model_desc == "gnb":
        return {}
    else:
        raise ValueError(f"Invalid value of 'model_desc' -- must be one of the following 'knn', 'dectree', 'gnb'; but not '{model_desc}'")


def compute_export_conformal_results(expl_feasibility, sparsity_counterfactual, sparsity_expl, overlap_expl_counterfactual):
    # Compute final statistics
    expl_feasibility_mean, expl_feasibility_var  = np.mean(expl_feasibility), np.var(expl_feasibility)
    sparsity_counterfactual_mean, sparsity_counterfactual_var = np.mean(sparsity_counterfactual), np.var(sparsity_counterfactual)
    sparsity_expl_mean, sparsity_expl_var = np.mean(sparsity_expl), np.var(sparsity_expl)
    overlap_expl_counterfactual_mean, overlap_expl_counterfactual_var = np.mean(overlap_expl_counterfactual), np.var(overlap_expl_counterfactual)

    # Export
    print(f"Expl feasibility: {expl_feasibility_mean} \pm {expl_feasibility_var}")
    print(f"Counterfactual sparsity: {sparsity_counterfactual_mean} \pm {sparsity_counterfactual_var}")
    print(f"Expl sparsity: {sparsity_expl_mean} \pm {sparsity_expl_var}")
    print(f"Overlap expl vs. counterfactual: {overlap_expl_counterfactual_mean} \pm {overlap_expl_counterfactual_var}")
   
    # LaTeX export
    print(f"${np.round(expl_feasibility_mean, 2)} \pm {np.round(expl_feasibility_var, 2)}$ & ${np.round(sparsity_expl_mean, 2)} \pm {np.round(sparsity_expl_var, 2)}$ & ${np.round(sparsity_counterfactual_mean, 2)} \pm {np.round(sparsity_counterfactual_var, 2)}$")

def compute_export_cnformal_perturbed_features_recovery_results(perturbed_features_recovery_counterfactual, perturbed_features_recovery_expl):
    # Compute final statistics
    perturbed_features_recovery_counterfactual_mean, perturbed_features_recovery_counterfactual_var = np.mean(perturbed_features_recovery_counterfactual), np.var(perturbed_features_recovery_counterfactual)
    perturbed_features_recovery_expl_mean, perturbed_features_recovery_expl_var = np.mean(perturbed_features_recovery_expl), np.var(perturbed_features_recovery_expl)
    
    # Export
    print(f"Perturbed features recovery counterfactual: {perturbed_features_recovery_counterfactual_mean} \pm {perturbed_features_recovery_counterfactual_var}")
    print(f"Perturbed features recovery expl: {perturbed_features_recovery_expl_mean} \pm {perturbed_features_recovery_expl_var}")

    # LaTeX export
    print(f"${np.round(perturbed_features_recovery_expl_mean, 2)} \pm {np.round(perturbed_features_recovery_expl_var, 2)}$ & ${np.round(perturbed_features_recovery_counterfactual_mean, 2)} \pm {np.round(perturbed_features_recovery_counterfactual_var, 2)}$")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: <dataset> <model> <local-explanation-model>")
        os._exit(1)

    # Specifications (provided as an input by the user)
    data_desc = sys.argv[1]
    model_desc = sys.argv[2]
    local_model_desc = sys.argv[3]

    # Load data
    X, y = load_data(data_desc)
    print(X.shape)

    # Results/Statistics
    expl_feasibility = [];expl_feasibility_perturbed = []
    sparsity_counterfactual = [];sparsity_counterfactual_perturbed = []
    sparsity_expl = [];sparsity_expl_perturbed = []
    overlap_expl_counterfactual = [];overlap_expl_counterfactual_perturbed = []
    perturbed_features_recovery_counterfactual = []
    perturbed_features_recovery_expl = []

    # In case of an extremly large majority class, perform simple downsampling
    if data_desc == "t21":
        rus = RandomUnderSampler()
        X, y = rus.fit_resample(X, y)

    # K-Fold
    for train_index, test_index in KFold(n_splits=n_folds, shuffle=True, random_state=None).split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # If necessary (in case of an highly imbalanced data set), apply Synthetic Minority Over-sampling Technique (SMOTE)
        if data_desc == "flip":
            sm = SMOTE(k_neighbors=1)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            X_test, y_test = sm.fit_resample(X_test, y_test)

        # Hyperparameter tuning
        model_search = ConformalRejectOptionGridSearchCV(model_class=get_model(model_desc), parameter_grid=get_model_parameters(model_desc), rejection_thresholds=reject_thresholds)
        best_params = model_search.fit(X_train, y_train)


        # Split training set into train and calibtration set (calibration set is needed for conformal prediction)
        X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.2)

        # Fit & evaluate model and reject option
        model = get_model(model_desc)(**best_params["model_params"])
        model.fit(X_train, y_train)
        print(f"Model score: {model.score(X_train, y_train)}, {model.score(X_test, y_test)}")

        conformal_model = ConformalPredictionClassifier(MyClassifierSklearnWrapper(model))
        conformal_model.fit(X_calib, y_calib)
        print(f"Conformal predictor score: {conformal_model.score(X_train, y_train)}, {conformal_model.score(X_test, y_test)}")

        print(f'Rejection threshold: {best_params["rejection_threshold"]}')
        reject_option = ConformalPredictionClassifierRejectOption(conformal_model, threshold=best_params["rejection_threshold"])

        explanator = Explanation(reject_option, model_desc=local_model_desc)

        # Select random subset of features which are going to be perturbed
        perturbed_features_idx = select_random_feature_subset(X_train.shape[1])
        print(f"Perturbed features: {perturbed_features_idx}")

        # For each sample in the test set, check if it is rejected
        y_rejects = []
        for i in range(X_test.shape[0]):
            x = X_test[i,:]
            if reject_option(x):
                y_rejects.append(i)
        print(f"{len(y_rejects)}/{X_test.shape[0]} are rejected")
        
        # Compute explanations for all rejected test samples
        expl_feasibility_ = []
        for idx in y_rejects:
            try:
                x_orig = X_test[idx, :]

                expl = explanator.compute_explanation(x_orig)
                if expl is None:    # Was computing an explanation successful?
                    expl_feasibility_.append(0)
                    continue
                else:
                    expl_feasibility_.append(1. / len(y_rejects))

                xcf = expl["counterfactual"]
                delta_cf = np.abs(xcf - x_orig)     # Feature importance according to counterfactual

                feat_imp = expl["local_feature_importance"] # Feature importance of local approximation

                # Sparsity -- i.e. complexity of the explanation
                sparsity_counterfactual.append(evaluate_sparsity(xcf, x_orig))
                sparsity_expl.append(evaluate_sparsity_ex(feat_imp))

                # Feature overlap between different explanations
                overlap_expl_counterfactual.append(evaluate_featureoverlap(xcf, feat_imp, x_orig))
            except Exception as ex:
                print(ex)
        expl_feasibility.append(np.sum(expl_feasibility_))

        # Find all samples in the test set that are rejected because of the perturbation
        X_test = apply_perturbation(X_test, perturbed_features_idx)  # Apply perturbation
        
        y_rejects_due_to_perturbations = []
        for i in range(X_test.shape[0]):    # Check which samples are now rejected
            x = X_test[i,:]
            if reject_option(x) and i not in y_rejects:
                y_rejects_due_to_perturbations.append(i)
        print(f"{len(y_rejects_due_to_perturbations)}/{X_test.shape[0]} are rejected due to perturbations")

        # Compute explanations for all rejected test samples
        expl_feasibility_perturbed_ = []
        for idx in y_rejects_due_to_perturbations:
            try:
                x_orig = X_test[idx, :]

                expl = explanator.compute_explanation(x_orig)
                if expl is None:    # Was computing an explanation successful?
                    expl_feasibility_perturbed_.append(0)
                    continue
                else:
                    expl_feasibility_perturbed_.append(1. / len(y_rejects_due_to_perturbations))

                xcf = expl["counterfactual"]
                delta_cf = np.abs(xcf - x_orig)     # Feature importance according to counterfactual

                feat_imp = expl["local_feature_importance"] # Feature importance of local approximation

                # Evaluation of feature importances
                # Are perturbed features recovered?
                perturbed_features_recovery_counterfactual.append(evaluate_perturbed_features_recovery(xcf, x_orig, perturbed_features_idx))
                perturbed_features_recovery_expl.append(evaluate_perturbed_features_recovery_ex(feat_imp, perturbed_features_idx))

                # Sparsity -- i.e. complexity of the explanation
                sparsity_counterfactual_perturbed.append(evaluate_sparsity(xcf, x_orig))
                sparsity_expl_perturbed.append(evaluate_sparsity_ex(feat_imp))

                # Feature overlap between different explanations
                overlap_expl_counterfactual_perturbed.append(evaluate_featureoverlap(xcf, feat_imp, x_orig))
            except Exception as ex:
                print(ex)
        expl_feasibility_perturbed.append(np.sum(expl_feasibility_perturbed_))

    # Compute and export final statistics
    compute_export_conformal_results(expl_feasibility, sparsity_counterfactual, sparsity_expl, overlap_expl_counterfactual)
    
    print("Perturbed features:")
    compute_export_conformal_results(expl_feasibility_perturbed, sparsity_counterfactual_perturbed, sparsity_expl_perturbed, overlap_expl_counterfactual_perturbed)
    compute_export_cnformal_perturbed_features_recovery_results(perturbed_features_recovery_counterfactual, perturbed_features_recovery_expl)
