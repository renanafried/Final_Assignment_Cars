import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from car_data_prep import prepare_data
import pickle

data = pd.read_csv("dataset.csv")
df = data.copy()

train_data1, test_data = train_test_split(df, test_size=0.2, random_state=0)
train_data1 = prepare_data(train_data1)
test_data = prepare_data(test_data)

# Separate majority and minority classes
df_majority = train_data1[train_data1['Year'] >= 2000]  # Example for majority years
df_minority = train_data1[train_data1['Year'] < 2000]   # Example for minority years

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                  replace=True,     # Sample with replacement
                                  n_samples=len(df_majority),  # Match number of majority class
                                  random_state=0)  # For reproducibility

# Combine majority class with upsampled minority class
train_data= pd.concat([df_majority, df_minority_upsampled])

frequent_value = train_data['capacity_Engine'].mode()[0]

#filling missing values and outliers with the mode value (from train data) for the engine capacity 
train_data['capacity_Engine'] = train_data['capacity_Engine'].fillna(frequent_value)
# Replace outliers for train data
train_data.loc[(train_data['capacity_Engine'] < 900) | (train_data['capacity_Engine'] > 4000), 'capacity_Engine'] = frequent_value
test_data['capacity_Engine'] = test_data['capacity_Engine'].fillna(frequent_value)
# Replace outliers for test data
test_data.loc[(test_data['capacity_Engine'] < 900) | (test_data['capacity_Engine'] > 4000), 'capacity_Engine'] = frequent_value

# Separate the features and target for training data
X_train = train_data.drop(['Price'], axis=1)
y_train = train_data['Price']

# Separate the features and target for test data
X_test = test_data.drop(['Price'], axis=1)
y_test = test_data['Price']

# Define categorical and numerical columns
categorical_columns = ['manufactor','model','Gear','Engine_type','Prev_ownership','Curr_ownership','Color']
numerical_columns = ['Year', 'Hand','capacity_Engine','Pic_num','Description','is_reposted']

# Create the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

# Create the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('E', ElasticNet())
])

# Define the parameter grid for hyperparameter search
param_grid = {
    'E__alpha': [0.05, 0.06, 0.07, 0.09, 0.1, 0.5, 1.0, 5.0, 5.5],
    'E__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Create the GridSearchCV object with 10-fold cross-validation
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10)

# Fit the GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict on the test data
y_pred = best_model.predict(X_test)
pickle.dump(best_model, open("trained_model.pkl","wb"))
