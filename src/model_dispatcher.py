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