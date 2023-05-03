import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, StratifiedKFold

import config
import module
import model_dispatcher


def run_cv(model, data):

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

    # run cv
    cv_scores= cross_validate(
        estimator= pipeline, 
        scoring= ['f1', 'accuracy'],
        X= data.drop(columns= label),
        y= data[label].values.ravel(), 
        cv= fold, 
        n_jobs= -1, 
        return_train_score= True
    )

    # create a df with scores
    # add model name and fold as new columns
    cv_scores_df= pd.DataFrame(cv_scores).assign(
        model= model, 
        fold= list(range(3)),
    )
        
    # return result df
    return cv_scores_df


if __name__ == '__main__':

    # load data
    # we load data outside of the run_cv function
    # so that we only need to lead data once for all models    
    df_train= pd.read_csv(config.TRAIN_SET)
    
    # list to contain resulting dfs
    results= []

    # loop through all models
    for model in model_dispatcher.models.keys():
        
        # run cv, save resulting df 
        result= run_cv(
            model= model, 
            data= df_train,
        )

        # append result to list
        results.append(result)

    # concat all dfs in results
    result_df = pd.concat(results)

    # save the result to output
    result_df.to_csv(config.TRAIN_RESULT, index= False)

    # print result
    print(result_df)
