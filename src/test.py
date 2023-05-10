import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score

import config
import module
import model_dispatcher


def run_test(model, data_train, data_test):

    # clean df
    unused= [
        'EmployeeCount',
        'EmployeeNumber',
        'StandardHours', 
        'Over18',
    ]
    data_train= module.clean_df(data_train, unused)
    data_test= module.clean_df(data_test, unused)

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
    num= [col for col in data_train.columns if col not in label + cat + num_root]

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

    # specify train and test set
    X_train= data_train.drop(columns= label)
    X_test= data_test.drop(columns= label)
    y_train= data_train[label].values.ravel()
    y_test= data_test[label].values.ravel()

    # fit pipeline to full train
    pipeline.fit(X_train, y_train)
     
    # predict
    y_pred= pipeline.predict(X_test)

    return pipeline, y_test, y_pred


if __name__ == '__main__':

    # load data
    df_train= pd.read_csv(config.TRAIN_SET)
    df_test= pd.read_csv(config.TEST_SET)
    
    # list to contain resulting dfs
    y_tests= []
    y_preds= []
    accs= []
    f1s= []

    models= [
        'logres_tuned_bayes',
        'logres_tuned_random',
        'ada_tuned_bayes',
        'sgd_tuned_random',
        'xgb_tuned_bayes',
    ]

    # loop through all models
    for model in models:
        
        # run test  
        clf, y_test, y_pred= run_test(
            model= model, 
            data_train= df_train, 
            data_test= df_test
        )

        # get acc and f1 score
        acc= accuracy_score(y_test, y_pred)
        f1= f1_score(y_test, y_pred)

        # append result to list
        y_tests.append(y_test)
        y_preds.append(y_pred)
        accs.append(acc)
        f1s.append(f1)

        # save model
        filename= f'{model}.sav'
        filepath= config.MODEL_DIR / filename 
        joblib.dump(clf, filepath)

    # concat all dfs in results
    result_df = pd.DataFrame(
        data= [accs, f1s], 
        columns= models, 
        index= ['test_accuracy', 'test_f1']
    ).transpose()

    # save the result to output
    result_df.to_csv(config.TEST_RESULT, index= True)

    # save y_pred and y_test
    import pickle
    # list of lists containing model, y_test and y_pred
    pred= [models, y_tests, y_preds]
    with open(config.TEST_PRED, 'wb') as f:
        pickle.dump(pred, f)

    # print result
    print(result_df)
