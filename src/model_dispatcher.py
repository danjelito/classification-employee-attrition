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


models = {
    "logres": LogisticRegression(),
    "sgd": SGDClassifier(),
    "svc": SVC(),
    "knn": KNeighborsClassifier(),
    "dt": DecisionTreeClassifier(),
    "gp": GaussianProcessClassifier(),
    "rf": RandomForestClassifier(),
    "xgb": XGBClassifier(),
    "lgb": LGBMClassifier(),
}

params = {
    "logres": {
        'prediction__model__penalty': ['l1', 'l2'],
        'prediction__model__C': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.75],
        'prediction__model__solver': ['liblinear'],
        'prediction__model__max_iter': [1000],
    },
    'sgd': {
        'prediction__model__penalty': ['l1', 'l2', 'elasticnet'],
        'prediction__model__alpha': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.75],
        'prediction__model__max_iter': [1000],
        # 'prediction__model__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'prediction__model__early_stopping': [True, False],
    }, 
    'xgb': {
        'prediction__model__max_depth': range(1, 7),
        'prediction__model__min_child_weight': range(0, 5),
        'prediction__model__gamma': range(0, 5),
        'prediction__model__max_delta_step': range(0, 11),
    },
}