import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import config
import module
import model_dispatcher


def run_randomized_search(model, data):

    # clean df
    unused= [
        'EmployeeCount',
        'EmployeeNumber',
        'StandardHours', 
        'Over18',
    ]
    data= module.clean_df(data, unused)

    # label
    label= ['Attrition']

    # categorical features
    cat= [
        'BusinessTravel',
        'Department',
        'EducationField',
        'Gender',
        'JobRole',
        'MaritalStatus',
        'OverTime',
    ]

    # numerical features that are long tailed
    num_root= [
        'DistanceFromHome',
        'MonthlyIncome',
        'NumCompaniesWorked',
        'PercentSalaryHike',
        'TotalWorkingYears',
        'YearsAtCompany',
        'YearsSinceLastPromotion',
    ]

    # numerical features
    num= [col for col in data.columns if col not in label + cat + num_root]

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
    prediction= Pipeline([
        ('model', model_dispatcher.models[model])
    ])

    # combine preprocessing and prediction pipeline
    pipeline= Pipeline([
        ('preprocessing', preprocessing),
        ('prediction', prediction)
    ])

    # initialize stratified kfold
    fold= StratifiedKFold(n_splits= 3, shuffle= True, random_state= config.RANDOM_STATE)

    # initialize grid search cv
    search= GridSearchCV(
        estimator= pipeline, 
        # get param distribution from model_dispatcher
        # according to model
        param_grid= model_dispatcher.params[model], 
        scoring= 'f1', 
        n_jobs= -1, 
        cv= fold, 
        verbose= 0
    )

    # run randomized search 
    search.fit(
        X= data.drop(columns= label),
        y= data[label].values.ravel(),
    )

    # get best estimator, param and score
    best_estimator= search.best_estimator_
    best_estimator_params= search.best_params_
    best_estimator_score= search.best_score_

    # print result
    print(f'Best params: {best_estimator_params}')
    print(f'Score: {best_estimator_score}')


if __name__ == '__main__':

    # load data
    # we load data outside of the run_cv function
    # so that we only need to lead data once for all models    
    df_train= pd.read_csv(config.TRAIN_SET)
    
    for model in [
        'logres', 
        'sgd', 
        # 'xgb'
    ]:
        
        # print model name and separator
        print(f'\n{"".join(["-"] * 50)}')
        print(f'{type(model_dispatcher.models[model]).__name__}')
        
        # run search 
        run_randomized_search(
            model= model, 
            data= df_train,
        )