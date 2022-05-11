import warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from ceml.sklearn import generate_counterfactual


class Explanation():
    def __init__(self, reject_option, model_desc="dectree", tree_max_depth=3, regularization_strength=1., num_samples=100, normal_loc=0., normal_scale=1., **kwds):
        self.reject_option = reject_option
        self.model_desc = model_desc
        self.tree_max_depth = tree_max_depth
        self.C = regularization_strength
        self.num_samples = num_samples
        self.normal_loc = normal_loc
        self.normal_scale = normal_scale

        super().__init__(**kwds)

    def __sample(self, x_orig):
        return x_orig + np.random.normal(loc=self.normal_loc, scale=self.normal_scale, size=x_orig.shape)

    def __fit_local_approximatation(self, x_orig):
        # Sample around x_orig
        X = [x_orig] + [self.__sample(x_orig) for _ in range(self.num_samples)]

        # Label samples according the output of the reject option
        y = [self.reject_option(x) for x in X]

        X = np.array(X)
        y = np.array(y)

        # Fit decision tree to labeled data set
        model = None
        if self.model_desc == "dectree":
            model = DecisionTreeClassifier(max_depth=self.tree_max_depth)
        elif self.model_desc == "logreg":
            model = LogisticRegression(penalty="l1", C=self.C, solver="saga", multi_class="multinomial")
        else:
            raise ValueError(f"Invalid value of 'model_desc' -- must be either 'dectree' or 'logreg' but not '{self.model_desc}'")

        model.fit(X, y.ravel())

        return model

    def compute_explanation(self, x_orig, features_whitelist=None):
        # Fit a local (simple) approximation of the model around x_orig
        model = self.__fit_local_approximatation(x_orig)

        # Compute a counterfactual explanation of x_orig under the local approximation
        if model.predict(x_orig.reshape(1, -1)) == 0:  # Is sample orignally rejected? If not, computing an explanation does not make much sense!
            return None
        xcf, _, _ = generate_counterfactual(model, x=x_orig, y_target=0, features_whitelist=features_whitelist, regularization="l1", return_as_dict=False)
        
        # Compute feature importances of the local approximation
        local_feature_importance = None
        if isinstance(model, LogisticRegression):
            local_feature_importance = (model.coef_ / (np.sum(model.coef_))).flatten()
        elif isinstance(model, DecisionTreeClassifier):
            local_feature_importance = model.tree_.compute_feature_importances(normalize=True).flatten()
        else:
            warnings.warn(f"Computation of feature importances for {type(model)} is not supported.")

        return {"counterfactual": xcf, "local_feature_importance": local_feature_importance}
