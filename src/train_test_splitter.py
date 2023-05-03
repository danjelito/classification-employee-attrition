import pandas as pd
import numpy as np
from pathlib import Path

import config

df= pd.read_csv(config.OG_DATASET)

suspected_important_feats= [
    'JobRole', 
    'BusinessTravel', 
    'OverTime', 
    'Age'
]

# we will use suspected features above
# to stratify train, val and test
# we do this because we suspect that these columns
# are important
df= (df
    .assign(
        AgeBin= lambda df_: pd.cut(df_['Age'], bins= 3), 
        TempFeats= lambda df_: df_['JobRole'].astype(str) + '_' +\
            df_['BusinessTravel'].astype(str) + '_' +\
            df_['OverTime'].astype(str) + '_' +\
            df_['AgeBin'].astype(str)
    )
    .drop(columns= ['AgeBin'])
)

# list of tempfeats that only occurs < 30
indexes= (df
    .loc[:, 'TempFeats']
    .value_counts()
    .loc[lambda x: x < 30]
    .index
)

# create a dictionary 
mapping= dict(zip(indexes, ['Other'] * len(indexes)))

# replace tempfeats that only occurs < 30
# with 'Other'
df= (df
    .assign(TempFeats= df['TempFeats'].replace(mapping))     
)

# create train test split
# get 15 samples per tempfeats
# as test set
df_test= (df
    .groupby('TempFeats', group_keys= False)
    .apply(lambda x: x.sample(15, random_state= config.RANDOM_STATE))
)

# the samples that are not in test set belongs to train set
df_train= (df
    # merge df with df_test
    .merge(
        df_test, 
        left_index= True, 
        right_index= True, 
        how= 'left', 
        suffixes= (None, '_test')
    )
    # get rows where tempfeats_test is na
    # meaning that it is not in test set
    .loc[lambda df_: df_['TempFeats_test'].isna()]
    # drop columns from right df
    .loc[:, :'TempFeats']
)    

# drop tempfeats col
df_train= df_train.drop(columns= 'TempFeats')
df_test= df_test.drop(columns= 'TempFeats')

# assert that len df = len train + len test
assert len(df) == (len(df_train) + len(df_test))

# assert that columns of train and test are the same
assert (df_test.columns == df_train.columns).all()

# save train and test set
df_train.to_csv(config.TRAIN_SET, index= False)
df_test.to_csv(config.TEST_SET, index= False)
