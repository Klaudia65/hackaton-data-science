import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Path Constants
DATA_DIR = "data/data.csv"
# Data Loading
df_data = pd.read_csv(DATA_DIR)

# Create directories if they don't exist
os.makedirs('/visualization/distribution/with0', exist_ok=True)
os.makedirs('/visualization/bivariate/with0', exist_ok=True)

#-------IMPUTATION--------

#distribution of the features with the replacement of NaN with mean 
#for feature with <10% of missing data

i=0
for col in df_data.columns:
    if df_data[col].isnull().mean() < 0.1:
        print("Feature : ", col, " has ", df_data[col].isnull().mean()*100, "% of missing data")
        df_data[col].fillna(df_data[col].mean(), inplace=True)

df_data40 = df_data[df_data.columns[df_data.isnull().mean() < 0.4]]

#df_data40.to_csv('data/data40.csv', index=False)

# Imputation for other features with more than 10% of missing data
# with iterative imputation
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.datasets import make_regression

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

df_sample = df_data40

# Appliquer l'imputation avec les paramètres optimisés
imputer = IterativeImputer(
    estimator=Ridge(),#Ridge ou BayesianRidge
    max_iter=10,
    random_state=0,
    n_nearest_features=100
)

X_imputed = imputer.fit_transform(df_sample)

X_imputed_df = pd.DataFrame(X_imputed, columns=df_sample.columns)

# save
X_imputed_df.to_csv('data/imputed_sample_ridge.csv', index=False)


