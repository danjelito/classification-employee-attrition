from sklearn.preprocessing import FunctionTransformer
import numpy as np

def clean_df(input_df, list_cols_to_drop):
    output_df= (input_df
        # replace space with _ if any
        .rename(columns= lambda c: c.replace(' ', '_'))
        # drop unused columns
        .drop(columns= list_cols_to_drop)          
        # map label to 0 and 1
        .assign(
            Attrition= lambda df_: df_['Attrition'].map({'Yes': 1, 'No': 0})
        )
    )
    return output_df

root_transformer= FunctionTransformer(
    func= np.sqrt, 
    inverse_func= np.square, 
    feature_names_out= 'one-to-one'
)

# unused columns
unused= [
    'EmployeeCount',
    'EmployeeNumber',
    'StandardHours', 
    'Over18',
]

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
num= [
    'Age',
    'DailyRate',
    'Education',
    'EnvironmentSatisfaction',
    'HourlyRate',
    'JobInvolvement',
    'JobLevel',
    'JobSatisfaction',
    'MonthlyRate',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'StockOptionLevel',
    'TrainingTimesLastYear',
    'WorkLifeBalance',
    'YearsInCurrentRole',
    'YearsWithCurrManager'
]