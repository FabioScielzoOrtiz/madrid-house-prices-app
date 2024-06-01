import polars as pl
from PyMachineLearning.preprocessing import encoder, imputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import pickle
import pandas as pd

#########################################################################################################################
# Data-Processing
#########################################################################################################################

data_path = r'C:\Users\fscielzo\Documents\Docker-Apps\madrid-house-prices-app\data\madrid_houses.csv'
madrid_houses_df = pl.read_csv(data_path)
variables_to_remove = ['', 'id', 'district', 'neighborhood']
madrid_houses_df = madrid_houses_df.select(pl.exclude(variables_to_remove))

unique_values = madrid_houses_df['house_type'].unique().to_list()
unique_values = [str(x) for x in unique_values]
new_values = [f'House type {i}' for i in unique_values]
replace_dict = dict(zip(unique_values, new_values))
madrid_houses_df = madrid_houses_df.with_columns(madrid_houses_df['house_type'].cast(str))
madrid_houses_df = madrid_houses_df.with_columns(pl.col('house_type').replace(replace_dict))

###############################################################################################################

response = 'buy_price'
quant_predictors = ['sq_mt_built', 'n_rooms',  'n_bathrooms',  'n_floors',  'sq_mt_allotment',  'floor']
cat_predictors = [x for x in madrid_houses_df.columns if x not in quant_predictors + [response]]
predictors = quant_predictors + cat_predictors

quant_pipeline = Pipeline([
    ('imputer', imputer(method='simple_mean'))
    ])

cat_pipeline = Pipeline([
    ('encoder', encoder(method='ordinal')), 
    ('imputer', imputer(method='simple_most_frequent'))
    ])

quant_cat_processing = ColumnTransformer(transformers=[('quant', quant_pipeline, quant_predictors),
                                                       ('cat', cat_pipeline, cat_predictors)])

pipeline = Pipeline([
        ('preprocessing', quant_cat_processing),
        ('LGB', LGBMRegressor(random_state=123, verbose=-1)) 
        ])

X = madrid_houses_df[predictors].to_pandas()
Y = madrid_houses_df[response].to_pandas()

pipeline.fit(X=X, y=Y)

saving_path = r'C:\Users\fscielzo\Documents\Docker-Apps\madrid-house-prices-app\model\model.pkl'
with open(saving_path, 'wb') as file:
    pickle.dump(pipeline, file)

#########################################################################################################################

def format_dtypes(dtype):
    if dtype in ['int64', 'float64']:
        return 'numerical'
    elif dtype == 'bool':
        return 'boolean'
    else:
        return 'string'
    
features_dtypes = dict(X.dtypes)
features_dtypes = {feature: format_dtypes(dtype) for feature, dtype in features_dtypes.items()}
features_metadata = {}
for col in X.columns:
    if features_dtypes[col] == 'boolean': # boolean case
        features_metadata[col] = {'dtype': features_dtypes[col],
                                  'unique_values': [str(x) for x in X[col].unique()]}
    elif features_dtypes[col] == 'numerical': # numerical case
        features_metadata[col] = {'dtype': features_dtypes[col],
                                  'min': X[col].min(),
                                  'max': X[col].max()}        
    elif features_dtypes[col] == 'string': # string case
        features_metadata[col] = {'dtype': features_dtypes[col],
                                  'unique_values': X[col].unique()}        
        
saving_path = r'C:\Users\fscielzo\Documents\Docker-Apps\madrid-house-prices-app\model\features_metadata.pkl'
with open(saving_path, 'wb') as file:
    pickle.dump(features_metadata, file)
print(features_metadata)

#########################################################################################################################

#########################################################################################################################



