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

from sklearn.preprocessing import FunctionTransformer
import numpy as np
root_transformer= FunctionTransformer(func= np.sqrt, inverse_func= np.square, feature_names_out= 'one-to-one')