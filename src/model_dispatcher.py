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

    # tuned model by gridd search
    "logres_tuned_grid": LogisticRegression(),
    "sgd_tuned_grid": SGDClassifier(),
    "xgb_tuned_grid": XGBClassifier(),
    
    # tuned model by bayesian optimization
    "logres_tuned_grid": LogisticRegression(),
    "sgd_tuned_grid": SGDClassifier(),
    "xgb_tuned_grid": XGBClassifier(),
}

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
        # try different lr which needs eta0 parameter
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