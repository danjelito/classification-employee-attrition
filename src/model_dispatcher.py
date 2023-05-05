from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import numpy as np

from skopt import space

models = {
    "logres": LogisticRegression(),
    "sgd": SGDClassifier(),
    "svc": SVC(),
    "knn": KNeighborsClassifier(),
    "dt": DecisionTreeClassifier(),
    "gp": GaussianProcessClassifier(),
    "rf": RandomForestClassifier(),
    "ada": AdaBoostClassifier(),
    "gb": GradientBoostingClassifier(),
    "xgb": XGBClassifier(),
    "lgb": LGBMClassifier(),

    # tuned model by randomized search
    "logres_tuned_random": LogisticRegression(**{
        'solver': 'liblinear', 
        'penalty': 'l2', 
        'max_iter': 1000, 
        'C': 0.0014
    }),
    "sgd_tuned_random": SGDClassifier(**{
        'penalty': 'l2', 
        'max_iter': 1000, 
        'learning_rate': 'invscaling', 
        'l1_ratio': 1.0, 
        'eta0': 0.0038, 
        'early_stopping': True, 
        'alpha': 0.0712
    }),
    "xgb_tuned_random": XGBClassifier(**{
        'n_estimators': 170, 
        'min_child_weight': 2, 
        'max_depth': 1, 
        'max_delta_step': 2, 
        'lambda': 0.889, 
        'gamma': 0, 'prediction__model__alpha': 0.0
    }),

    # # tuned model by bayesian optimization
    "logres_tuned_bayes": LogisticRegression(**{
        'penalty': 'l2', 
        'C': 0.0014295171076570674, 
        'max_iter': 1000, 
        'solver': 'liblinear'
    }),
    "sgd_tuned_bayes": SGDClassifier(**{
        'penalty': 'elasticnet', 
        'alpha': 0.00016380341478793747, 
        'max_iter': 1000, 
        'early_stopping': True, 
        'learning_rate': 'invscaling', 
        'eta0': 1.0255201315268034
    }),
    "xgb_tuned_bayes": XGBClassifier(**{
        'max_depth': 2, 
        'min_child_weight': 0, 
        'gamma': 0, 
        'max_delta_step': 0, 
        'n_estimators': 237, 
        'lambda': 10, 
        'alpha': 0, 
        'tree_method': 'hist'
    }),
}

# params per model for grid search and randomized search
params = {

    "logres": [
        # linear solver compatible with l1 and l2 penalty
        {
            'prediction__model__penalty': ['l1', 'l2'],
            'prediction__model__C': np.arange(0.0001, 0.1, 0.0001),
            'prediction__model__solver': ['liblinear'],
            'prediction__model__max_iter': [1000],
        },
        # try other solver, which only compatile with l2 penalty
        {
            'prediction__model__penalty': ['l2'],
            'prediction__model__C': np.arange(0.0001, 0.1, 0.0001),
            'prediction__model__solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
            'prediction__model__max_iter': [2000],
        },
    ],

    'sgd': [
        {
            'prediction__model__penalty': ['l1', 'l2', 'elasticnet'],
            'prediction__model__alpha': np.arange(0.0001, 0.1, 0.0001),
            'prediction__model__max_iter': [1000],
            'prediction__model__early_stopping': [True, False],
        },
        # try different learning rate, which needs eta0 parameter
        {
            'prediction__model__penalty': ['l1', 'l2', 'elasticnet'],
            'prediction__model__l1_ratio': np.linspace(0, 1, 20),
            'prediction__model__alpha': np.arange(0.0001, 0.1, 0.0001),
            'prediction__model__max_iter': [1000],
            'prediction__model__early_stopping': [True, False],
            'prediction__model__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'prediction__model__eta0': np.logspace(-4, 1, 20),
        }, 
    ],

    'xgb': {
        'prediction__model__max_depth': range(1, 7),
        'prediction__model__min_child_weight': range(0, 5),
        'prediction__model__gamma': range(0, 4),
        'prediction__model__max_delta_step': range(0, 6),
        'prediction__model__n_estimators': range(50, 200, 10),
        'prediction__model__lambda': np.linspace(0, 2, 10),
        'prediction__model__alpha': np.linspace(0, 2, 10),
    },
}

#param spaces and names for bayesian optimization
param_spaces = {

    'logres': [
        space.Categorical(['l2'], name= 'penalty'), 
        space.Real(0.00001, 1, prior="log-uniform", name= "C"),
        space.Categorical([1000], name= 'max_iter'), 
        space.Categorical(['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], name= 'solver'), 
    ],

    'sgd': [
        space.Categorical(['l1', 'l2', 'elasticnet'], name= 'penalty'), 
        space.Real(0.00001, 1, prior= 'log-uniform', name= 'alpha'),
        space.Categorical([1000], name= 'max_iter'), 
        space.Categorical([True, False], name= 'early_stopping'), 
        space.Categorical(['constant', 'invscaling', 'adaptive'], name= 'learning_rate'), 
        space.Real(0.00001, 10, prior= 'log-uniform', name= 'eta0'),
    ],

    'xgb': [
        space.Integer(1, 5, name= 'max_depth'),
        space.Integer(0, 10, name= 'min_child_weight'),
        space.Integer(0, 10, name= 'gamma'),
        space.Integer(0, 10, name= 'max_delta_step'),
        space.Integer(10, 300, name= 'n_estimators'),
        space.Integer(0, 10, name= 'lambda'),
        space.Integer(0, 10, name= 'alpha'),
        space.Categorical(['approx', 'hist', 'auto', 'exact'], name= 'tree_method'), 
    ],

}

param_names = {

    'logres': [
        'penalty',
        'C',
        'max_iter',
        'solver',
    ],

    'sgd': [
        'penalty', 
        'alpha', 
        'max_iter',
        'early_stopping',
        'learning_rate', 
        'eta0',
    ],

    'xgb': [
        'max_depth',
        'min_child_weight',
        'gamma',
        'max_delta_step',
        'n_estimators',
        'lambda',
        'alpha',
        'tree_method',
    ],

}
