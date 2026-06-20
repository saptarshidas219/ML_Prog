# import all the libraries: 
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error
import kagglehub
import os

# Load the data
path = kagglehub.dataset_download("camnugent/california-housing-prices")
print("Dataset downloaded to path:", path)
files = os.listdir(path)
print("Files available:", files)
csv_file = [f for f in files if f.endswith('.csv')][0] # Grabs the first CSV found
housing = pd.read_csv(os.path.join(path, csv_file))

#split the data:
housing['income cat'] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
spilt = StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=42)
for train_index,test_index in spilt.split(housing,housing["income cat"]):
    strat_train_set = housing.iloc[train_index]
    strat_test_set = housing.iloc[test_index]

housing = strat_train_set.copy()
#Seperate predictors & labels
housing_label = housing['median_house_value'] #seperate predictors
housing = housing.drop(columns = ['median_house_value'],axis=1 ) #seperate labels

# Seperate categorical & numerical columns
num_attributes = housing.drop(columns = ['ocean_proximity'], axis=1).columns.tolist()
cat_attributes = ['ocean_proximity']

# create pipelines:

# categorical Pipeline
num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

# Numerical Pipeline
cat_pipeline = Pipeline([
    ('onehat',OneHotEncoder(handle_unknown='ignore'))
])

#ConsolidatePipeline
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attributes),
    ("cat",cat_pipeline,cat_attributes)
])

# Transform the Data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_label)

# Decison Tree 
tree_reg = DecisionTreeClassifier()
tree_reg.fit(housing_prepared,housing_label)

# Random Forest
forest_reg = RandomForestClassifier()
forest_reg.fit(housing_prepared,housing_label)


# RMSE Accuracy
lin_rmse = root_mean_squared_error(housing_label,lin_reg,squared = False)
tree_rmse = root_mean_squared_error(housing_label,tree_reg,squared = False)
forest_rmse = root_mean_squared_error(housing_label,forest_reg,squared = False)


print(f"RMSE Linear Regression: {lin_rmse}")
print(f"RMSE Decision Tree Regression: {tree_rmse}")
print(f"RMSE Random Forest Regression: {forest_rmse}")

