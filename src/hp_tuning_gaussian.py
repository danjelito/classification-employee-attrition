import pandas as pd
import numpy as np
from functools import partial

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer

from skopt import gp_minimize

import model_dispatcher
import config
import module


def run_cv(
    params: list, 
    model: str, 
    param_names: list,
    data: pd.DataFrame
):
    """
    Cross validation with the given model, parameters and dataset

    Args:
        params (list): model parameters
        model (str): model to use, taken from model_dispatcher
        param_names (list): name of the model parameters
        data (pd.DataFrame): the dataset consisting of X and y

    Returns:
        int: negative mean f1 score across the folds
    """
    
    # convert params to dictionary 
    params= dict(zip(param_names, params))

    # clean df
    unused= module.unused
    data= module.clean_df(data, unused)

    # label
    label= module.label

    # categorical features
    cat= module.cat

    # numerical features that are long tailed
    num_root= module.num_root

    # numerical features
    num= module.num
    
    # create preprocessing pipeline
    cat_pipe= Pipeline([
        ('impute', SimpleImputer(strategy= 'constant', fill_value= 'NONE')), 
        ('ohe', OneHotEncoder(sparse_output= False, handle_unknown= 'ignore')), 
        ('scale', StandardScaler())
    ])
    num_pipe= Pipeline([
        ('impute', KNNImputer(n_neighbors= 3)), 
        ('scale', StandardScaler())
    ])
    num_root_pipe= Pipeline([
        ('impute', KNNImputer(n_neighbors= 3)), 
        ('transform', module.root_transformer),
        ('scale', StandardScaler()),
    ])
    preprocessing= ColumnTransformer([
        ('cat', cat_pipe, cat), 
        ('num', num_pipe, num), 
        ('num_root', num_root_pipe, num_root), 
    ])

    # create predition pipeline
    # we reset the model args to use **params
    prediction= Pipeline([
        ('model', model.set_params(**params))
    ])

    # combine preprocessing and prediction pipeline
    pipeline= Pipeline([
        ('preprocessing', preprocessing),
        ('prediction', prediction)
    ])

    # initialize stratified kfold
    fold= StratifiedKFold(n_splits= 3, shuffle= True, random_state= config.RANDOM_STATE)

    # get cv score
    f1_scores= cross_val_score(
        estimator= pipeline, 
        X= data.drop(columns= label),
        y= data[label].values.ravel(),
        cv= fold,
        scoring= 'f1', 
    )

    # return negative mean cv core
    return np.mean(f1_scores) * -1 

def optimize(
    model: str, 
    n_calls: int 
):
    """
    Use gp_minimize to minimize loss function
    given by function run_cv

    Args:
        model (str): model to use, taken from model_dispatcher
        n_calls (int): number of iterations
    """
    
    df_train= pd.read_csv(config.TRAIN_SET)

    # create a partial function
    # leave the param_space as the only remaining arg
    partial_cv= partial(
        run_cv, 
        model= model_dispatcher.models[model], # model will be called by run_cv function
        param_names= model_dispatcher.param_names[model], 
        data= df_train
    )

    # run gp minimize
    result= gp_minimize(
        func= partial_cv, # function to minimize
        dimensions= model_dispatcher.param_spaces[model],
        n_calls= n_calls,
        verbose= 0, 
        n_jobs= -1,
        random_state= config.RANDOM_STATE,
    )

    # print model name and separator
    print(f'\n{"".join(["-"] * 50)}')
    print(f'{type(model_dispatcher.models[model]).__name__}')

    # print best params
    best_params= dict(zip(model_dispatcher.param_names[model], result.x))
    print(f'\nBest params: {best_params}\n')
    

if __name__ == '__main__':

    # ignore user warning
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category= UserWarning)

        for model in [
            'logres', 
            'ada', 
            'sgd'
        ]:
            
            optimize(
                model= model,
                n_calls= 100,
            )
