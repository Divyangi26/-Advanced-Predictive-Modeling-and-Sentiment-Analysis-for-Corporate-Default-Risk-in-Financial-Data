import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LassoCV, LogisticRegression,RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import altair as alt

import tensorflow as tf
from tensorflow import keras

import nltk
from nltk.tokenize import sent_tokenize

from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline
from collections import Counter
import json


####DATA EXTRACTION AND CLEANING

#extracting bankruptcy data 
bankruptcy_data = pd.read_csv("data/BR1964_2019.csv")
bankruptcy_data = bankruptcy_data[bankruptcy_data['PERMNO'].notna()]
#reading in the compstat data
compstat_data = pd.read_csv("data/funda_2022.csv")
#cleaning compstat data
compstat_data = compstat_data[['cik','sale','at','gvkey','cusip','mkvalt','datadate','dlc','dltt','teq','act','lct','recch','apalch','re','oiadp','prcc_f','csho','ni']]


#calculating the necessary ratios 
compstat_data['debt_to_equity_ratio'] = (compstat_data['dlc']+compstat_data['dltt'])/compstat_data['teq']
compstat_data['roe'] = compstat_data['ni']/compstat_data['teq']
compstat_data['roa'] = compstat_data['ni']/compstat_data['at']
compstat_data['accounts_receivable'] = compstat_data['recch']
compstat_data['accounts_payable'] = compstat_data['apalch']
compstat_data['trailing_2_yr_revenues'] = compstat_data.groupby('gvkey')['sale'].rolling(window=2).mean().reset_index(0, drop=True)
compstat_data['working_capital'] = compstat_data['act'] - compstat_data['lct']
compstat_data['market_to_book'] = compstat_data['mkvalt']/compstat_data['at']
compstat_data['wc_ta'] = compstat_data['working_capital']/compstat_data['at']
compstat_data['re_ta'] = compstat_data['re']/compstat_data['at']
compstat_data['ebit_ta'] = compstat_data['oiadp']/compstat_data['at']
compstat_data['mv_tl'] = (compstat_data['prcc_f']*compstat_data['csho'])/compstat_data['at']
compstat_data['sales_ta'] = compstat_data['sale']/compstat_data['at']

compstat_data['atlman_score']= 1.2* compstat_data['wc_ta']+ 1.4*compstat_data['re_ta'] + 3.3 *compstat_data['ebit_ta'] + 0.6 * compstat_data['mv_tl'] + 0.99*compstat_data['sales_ta']
compstat_data['lagged_total_assets'] = compstat_data.groupby('gvkey')['at'].shift(1)
compstat_data['asset_turnover'] = compstat_data['sale'] / (abs(compstat_data['at']) + abs(compstat_data['lagged_total_assets']))
# Assuming 'your_dataset' is your DataFrame
compstat_data.replace([np.inf, -np.inf], np.nan, inplace=True)
compstat_data.fillna(0, inplace=True)
#computing the required variables
compstat_data['cusip'] =  compstat_data['cusip'].str[0:6]
compstat_data['datadate'] = pd.to_datetime(compstat_data['datadate'],errors='coerce')
compstat_data['year'] = compstat_data['datadate'].apply(lambda x:x.year)
#lagging the data to previous year results
compstat_data['datadate'] = compstat_data['year'].apply(lambda x:f'{x+1}-01-01')
compstat_data['datadate'] = pd.to_datetime(compstat_data['datadate'],errors='coerce')
compstat_data['year'] = compstat_data['datadate'].apply(lambda x:x.year)
#reading monthly stock market data
crsp_data = pd.read_csv("data/msf_1926_2022.csv",usecols=['CUSIP','PERMNO','date','PRC','SHROUT','RET','vwretd','BID','ASK'])

#cleaning crsp data and converting and storing the annual data into dsf data
crsp_data['date'] = pd.to_datetime(crsp_data['date'],errors='coerce')
crsp_data['year'] = crsp_data['date'].apply(lambda x:x.year)
crsp_data['CUSIP'] = crsp_data['CUSIP'].str[0:6]
crsp_data['SHROUT'] = crsp_data['SHROUT']*1000
crsp_data['Bid_Ask'] = crsp_data['BID'] - crsp_data['ASK']
crsp_data['E'] = abs(crsp_data['PRC'])*crsp_data['SHROUT']
crsp_data['RET'] = pd.to_numeric(crsp_data['RET'],errors='coerce')
crsp_data.dropna(inplace=True)
#data groupby
ann_ret = crsp_data.groupby(['CUSIP', 'year']).apply(lambda x: np.exp(np.sum(np.log(1 + x['RET']))))
sigma_e = crsp_data.groupby(['CUSIP', 'year'])['RET'].std() * np.sqrt(12)

#annual data
msf_data = crsp_data.groupby(['CUSIP', 'year']).first()
msf_data['ANNRET'] = ann_ret
msf_data['SIGMAE'] = sigma_e
msf_data.reset_index(inplace=True)
#lagging the data to previous year results
msf_data['date'] = msf_data['year'].apply(lambda x:f'{x+1}-01-01')
msf_data['date'] = pd.to_datetime(msf_data['date'],errors='coerce')
msf_data['year'] = msf_data['date'].apply(lambda x:x.year)
#converting column names to uppercase
compstat_data.columns = compstat_data.columns.str.upper()
msf_data.columns = msf_data.columns.str.upper()
#merge the compustat and monthly stock data
merged_data = pd.merge(msf_data, compstat_data, on=['CUSIP', 'YEAR'])
merged_data.drop(columns=['DATADATE'],inplace=True)
bankruptcy_data['bankruptcy_dt'] = pd.to_datetime(bankruptcy_data['bankruptcy_dt'],errors='coerce')
bankruptcy_data['Year'] = bankruptcy_data['bankruptcy_dt'].apply(lambda x:x.year)
bankruptcy_data.columns = bankruptcy_data.columns.str.upper()

# Merge the compustat, stock and bankruptcy data on PERMNO and year
merged_data_2 = pd.merge(merged_data, bankruptcy_data, how='left', on=['PERMNO', 'YEAR'])

# Create a new column 'bkr_indicator' with 1 for matches and 0 for non-matches
merged_data_2['bkr_indicator'] = merged_data_2['BANKRUPTCY_DT'].notnull().astype(int)

# Fill NaN values in case there are no matches
merged_data_2['bkr_indicator'].fillna(0, inplace=True)
merged_data_2.drop(columns=['BANKRUPTCY_DT'],inplace=True)

merged_data_2.dropna(inplace=True)

####FUNCTION FOR THE MACHINE LEARNING ALGORITHMS

explanatory_variables = ['MARKET_TO_BOOK', 'NI','ANNRET','DEBT_TO_EQUITY_RATIO','ROE', 'ROA','SIGMAE','ACCOUNTS_RECEIVABLE','ATLMAN_SCORE','ACCOUNTS_PAYABLE','TRAILING_2_YR_REVENUES','WORKING_CAPITAL','ASSET_TURNOVER']
plt.figure(figsize=(3, 2))
X = merged_data_2[explanatory_variables]
y = merged_data_2['bkr_indicator']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state=5)
# Plotting a histogram
plt.hist(y_train, bins='auto', color='blue', alpha=0.7, rwidth=0.85)

# Adding labels and title
plt.xlabel('Target Variable Values')
plt.ylabel('Frequency')
plt.title('Distribution of Target Variable (y_train)')

# Show the plot
plt.show()

plt.figure(figsize=(3, 2))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state=5)
# Apply RandomUnderSampler with a specific sampling strategy (adjust the ratio as needed)
undersampler = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
X_train, y_train = undersampler.fit_resample(X_train, y_train)
    # Apply SMOTE with a specific sampling strategy (adjust the ratio as needed)
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
plt.hist(y_train, bins='auto', color='blue', alpha=0.7, rwidth=0.85)

# Adding labels and title
plt.xlabel('Target Variable Values')
plt.ylabel('Frequency')
plt.title('Distribution of Target Variable (y_train)')

# Show the plot
plt.show()

#initialising  independent variables X and dependent variables y

def in_sample_data_initialise():
    X = merged_data_2[explanatory_variables]
    y = merged_data_2['bkr_indicator']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state=5)
    # Apply RandomUnderSampler with a specific sampling strategy (adjust the ratio as needed)
    undersampler = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)
        # Apply SMOTE with a specific sampling strategy (adjust the ratio as needed)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train,X_test,y_train,y_test 


def out_of_sample_initialise():
    X_train = merged_data_2[(merged_data_2['YEAR']>=1964) & (merged_data_2['YEAR']<=1990)][explanatory_variables]
    y_train = merged_data_2[(merged_data_2['YEAR']>=1964) & (merged_data_2['YEAR']<=1990)]['bkr_indicator']
    X_test = merged_data_2[(merged_data_2['YEAR']>=1991) & (merged_data_2['YEAR']<=2019)][explanatory_variables]
    y_test = merged_data_2[(merged_data_2['YEAR']>=1991) & (merged_data_2['YEAR']<=2019)]['bkr_indicator']

    # Apply RandomUnderSampler with a specific sampling strategy (adjust the ratio as needed)
    undersampler = RandomUnderSampler(sampling_strategy=0.4, random_state=42)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)
        # Apply SMOTE with a specific sampling strategy (adjust the ratio as needed)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)


    return X_train,y_train,X_test,y_test
def print_decile_defaults(model,X_test,y_pred):
    # Assuming you have trained the logistic regression model (model.fit(X_train, y_train))


    # Perform cross-validated prediction for out-of-sample data
    cv_default_probabilities_test = cross_val_predict(model, X_test, y_pred, method='predict_proba')[:,1]
    # Combine probabilities and true labels into a DataFrame for analysis
    predictions_df = pd.DataFrame({'Probability': cv_default_probabilities_test, 'Default': y_pred})


    # Assign deciles based on custom thresholds
    predictions_df['Decile'] = pd.qcut(predictions_df['Probability'],q=10, labels=None)

    # # Create deciles based on predicted probabilities
    # predictions_df['Decile'] = pd.qcut(predictions_df['Probability'], q=10, labels=False, duplicates='drop')

    # Filter the DataFrame to include only observations where y_test is 1
    defaults_df = predictions_df[predictions_df['Default'] == 1]

    # Compute the number and percentage of defaults in each decile
    decile_stats = defaults_df.groupby('Decile')['Default'].count()


    # Display the results
    print(decile_stats)



def print_roc_graph_auc_ks(y_test,y_pred):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Calculate KS statistic
    ks_statistic = max(tpr - fpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.9f}, KS = {ks_statistic:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def performance(y_test,y_pred):   
    misclassification_rate = 1-accuracy_score(y_test,y_pred)
    print(f"\n---------------Misclassification Rate: {misclassification_rate:.8f}----------------")
    print(f"\n---------------Accuracy Score: {accuracy_score(y_test,y_pred):.8f}-----------------\n")
    print_roc_graph_auc_ks(y_test,y_pred)


###MODEL-1 , LOGISTIC REGRESSION
###IN - SAMPLE
    
X_train,X_test,y_train,y_test = in_sample_data_initialise()
logistic_model = LogisticRegression()

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

performance(y_test,y_pred)


####Summary

print(classification_report(y_test,y_pred))
# Get the coefficients and feature names
coefficients = logistic_model.coef_[0]
feature_names = X_train.columns

# Create a DataFrame to display the coefficients
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Display the coefficients
print(coefficients_df)

####Out of the sample_testing
####METHOD 1

X_train,y_train,X_test,y_test = out_of_sample_initialise()
# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)


performance(y_test,y_pred)
# NOTE :  Here the last two deciles are with the top most probability and initial deciles are with least probability of default

print_decile_defaults(logistic_model,X_test,y_pred) 

####METHOD 2

y_pred_rolling = []
for year in range(1991, 2020):
    # Use data up to the current year for training
    X_train_rolling = merged_data_2[(merged_data_2['YEAR']>=1964) & (merged_data_2['YEAR'] <= year - 1)][explanatory_variables]
    y_train_rolling = merged_data_2[(merged_data_2['YEAR']>=1964) & (merged_data_2['YEAR'] <= year - 1)]['bkr_indicator']
    
    # Fit the model
    logistic_model.fit(X_train_rolling, y_train_rolling)
    
    # Make predictions for the current year
    X_current_year = merged_data_2[merged_data_2['YEAR'] == year][explanatory_variables]
    y_pred_2= logistic_model.predict(X_current_year)
    y_pred_rolling.append(y_pred_2)
result_array = np.concatenate(y_pred_rolling, axis=0)
accuracy_score(y_test,result_array)
roc_auc_score(y_test,result_array)


###METHOD 3


y_pred_fixed = []
for year in range(1991, 2020):
    # Use a fixed window for training
    X_train_fixed = merged_data_2[(merged_data_2['YEAR'] >= 1964) & (merged_data_2['YEAR'] <= year - 1)][explanatory_variables]
    y_train_fixed = merged_data_2[(merged_data_2['YEAR'] >= 1964) & (merged_data_2['YEAR'] <= year - 1)]['bkr_indicator']

    # Fit the model
    logistic_model.fit(X_train_fixed, y_train_fixed)
    
    # Make predictions for the current year
    X_current_year = merged_data_2[merged_data_2['YEAR'] == year][explanatory_variables]
    y_pred_3 = logistic_model.predict(X_current_year)
    y_pred_fixed.append(y_pred_3)
result_array_2 = np.concatenate(y_pred_fixed,axis=0)
accuracy_score(y_test,result_array_2)
roc_auc_score(y_test,result_array_2)


####MODEL - 2 , LASSO AND RIDGE REGRESSION
####POST LASSO MODEL

X_train,y_train,X_test,y_test = out_of_sample_initialise()
X_train_in_sample,X_test_in_sample,y_train_in_sample,y_test_in_sample = in_sample_data_initialise()
# Step 1: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_in_sample)
X_test_scaled = scaler.transform(X_test_in_sample)

# Step 2: Initial LASSO Model
lasso_model = LogisticRegression(penalty='l1', solver='liblinear')
lasso_model.fit(X_train_scaled, y_train_in_sample)

# Step 3: Hyperparameter Tuning for LASSO
lasso_cv_model = LassoCV(cv=5)
lasso_cv_model.fit(X_train_scaled, y_train_in_sample)
best_alpha_lasso = lasso_cv_model.alpha_

X_train_scaled_out_sample = scaler.fit_transform(X_train)
X_test_scaled_out_sample = scaler.transform(X_test)

# Step 4: Post LASSO Logistic Regression
# Use only the selected features by LASSO
X_train_lasso_selected = X_train_scaled_out_sample[:, lasso_model.coef_[0] != 0]
post_lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=1 / best_alpha_lasso)
post_lasso_model.fit(X_train_lasso_selected, y_train)

# Step 5: Out-of-Sample Prediction for LASSO
# Use the same selected features for test set
X_test_lasso_selected = X_test_scaled_out_sample[:, lasso_model.coef_[0] != 0]
y_pred_lasso = post_lasso_model.predict(X_test_lasso_selected)

performance(y_test,y_pred_lasso)
print_decile_defaults(post_lasso_model,X_test,y_pred_lasso)


####POST RIDGE MODEL

# Assuming X_train, X_test, y_train, y_test, X_train_scaled, and X_test_scaled are defined
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_in_sample)
X_test_scaled = scaler.transform(X_test_in_sample)
# Step 2: Initial Ridge Model
ridge_model = LogisticRegression(penalty='l2', solver='liblinear')
ridge_model.fit(X_train_scaled, y_train)

# Step 3: Hyperparameter Tuning for Ridge
ridge_cv_model = RidgeCV(cv=5)
ridge_cv_model.fit(X_train_scaled, y_train)
best_alpha_ridge = ridge_cv_model.alpha_

# Step 4: Post Ridge Logistic Regression
# Use only the selected features by RidgeCV
X_train_ridge_selected = X_train_scaled_out_sample[:, ridge_model.coef_[0] != 0]
post_ridge_model = LogisticRegression(penalty='l2', solver='liblinear', C=1 / ridge_cv_model.alpha_)
post_ridge_model.fit(X_train_ridge_selected, y_train)



# Step 5: Out-of-Sample Prediction for Ridge
# Use the same selected features for test set
X_test_ridge_selected = X_test_scaled_out_sample[:, ridge_model.coef_[0] != 0]
y_pred_ridge = post_ridge_model.predict(X_test_ridge_selected)

# Assess the performance
performance(y_test, y_pred_ridge)

print_decile_defaults(post_ridge_model,X_test,y_pred_ridge)


###MODEL - 4 : KNN Model

X_train,y_train,X_test,y_test = out_of_sample_initialise()
knn_model = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}  # Example values, adjust as needed
grid_search = GridSearchCV(knn_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']

# Step 3: Fit K-Nearest Neighbors model with the optimal K
knn_model_best = KNeighborsClassifier(n_neighbors=best_k)
knn_model_best.fit(X_train, y_train)

# Make predictions on the out-of-sample data
y_pred_knn = knn_model_best.predict(X_test)

performance(y_test,y_pred_knn)
best_k


###MODEL 5 AND 6
####RANDOM FOREST 

X_train,y_train,X_test,y_test = out_of_sample_initialise()


# Assuming X_train, y_train, X_test, y_test are defined

# Step 1: Run Random Forest classification model for out-of-sample
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Step 2: Calibrate hyperparameters (max depth, number of trees, and additional parameters) using GridSearchCV
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],  # Adjust as needed
    'n_estimators': [50, 100, 200, 300],  # Adjust as needed
    'min_samples_split': [2, 5, 10],      # Additional parameter
    'min_samples_leaf': [1, 2, 4],        # Additional parameter
    'max_features': ['auto', 'sqrt', 'log2'],  # Additional parameter
    'bootstrap': [True, False]            # Additional parameter
}


grid_search_rf = GridSearchCV(rf_model, param_grid, cv=5)
grid_search_rf.fit(X_train, y_train)

# Get the best hyperparameters
best_max_depth = grid_search_rf.best_params_['max_depth']
best_n_estimators = grid_search_rf.best_params_['n_estimators']

# Step 3: Fit Random Forest model with the optimal hyperparameters
rf_model_best = RandomForestClassifier(max_depth=best_max_depth, n_estimators=best_n_estimators, random_state=42)
rf_model_best.fit(X_train, y_train)

# Make predictions on the out-of-sample data
y_pred_rf = rf_model_best.predict(X_test)


performance(y_test, y_pred_rf)
print(grid_search_rf.best_params_)

###SURVIVAL RANDOM FOREST

# Assuming 'bkr_indicator' is a binary indicator (0 for censored, 1 for bankruptcy)
# and 'YEAR' is the time of the event or censoring
event_train = merged_data_2[(merged_data_2['YEAR'] >= 1964) & (merged_data_2['YEAR'] <= 1990)]['bkr_indicator'].astype(bool)
time_train = merged_data_2[(merged_data_2['YEAR'] >= 1964) & (merged_data_2['YEAR'] <= 1990)]['YEAR']

event_test = merged_data_2[(merged_data_2['YEAR'] >= 1991) & (merged_data_2['YEAR'] <= 2019)]['bkr_indicator'].astype(bool)
time_test = merged_data_2[(merged_data_2['YEAR'] >= 1991) & (merged_data_2['YEAR'] <= 2019)]['YEAR']

# Create structured arrays for training and testing sets
y_train = np.array(list(zip(event_train, time_train)), dtype=[('event', bool), ('time', int)])
y_test = np.array(list(zip(event_test, time_test)), dtype=[('event', bool), ('time', int)])

# Your X_train and X_test remain the same
X_train = merged_data_2[(merged_data_2['YEAR'] >= 1964) & (merged_data_2['YEAR'] <= 1990)][['NI','ANNRET','DEBT_TO_EQUITY_RATIO','ROE', 'ROA','SIGMAE','ACCOUNTS_RECEIVABLE','ATLMAN_SCORE','ACCOUNTS_PAYABLE','TRAILING_2_YR_REVENUES','WORKING_CAPITAL','ASSET_TURNOVER']]
X_test = merged_data_2[(merged_data_2['YEAR'] >= 1991) & (merged_data_2['YEAR'] <= 2019)][['NI','ANNRET','DEBT_TO_EQUITY_RATIO','ROE', 'ROA','SIGMAE','ACCOUNTS_RECEIVABLE','ATLMAN_SCORE','ACCOUNTS_PAYABLE','TRAILING_2_YR_REVENUES','WORKING_CAPITAL','ASSET_TURNOVER']]

from sksurv.metrics import concordance_index_censored
# Define the parameter grid for hyperparameter tuning
param_grid = {'n_estimators': [5,10,15,20]}  # You can adjust the values as needed

# Create the Survival Random Forest model
rsf = RandomSurvivalForest()

# Use GridSearchCV to find the optimal number of trees
grid_search = GridSearchCV(rsf, param_grid, cv=5, scoring=concordance_index_censored)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_n_estimators = grid_search.best_params_['n_estimators']

print(f"Optimal number of trees: {best_n_estimators}")

# Create the model with the optimal number of trees
rsf_optimal = RandomSurvivalForest(n_estimators=best_n_estimators)

# Fit the model on the entire training set
rsf_optimal.fit(X_train, y_train)

# Make predictions on the test set
prediction = rsf_optimal.predict(X_test)


def evaluate_survival_model(model, X, y):
    # Make predictions on the survival data
    prediction = model.predict(X)
    
    # Calculate the concordance index
    concordance_index = concordance_index_censored(y['event'], y['time'], prediction)
    
    print(f"Concordance Index: {concordance_index[0]:.4f}")


# Assuming X_test and y_test are defined for survival analysis
evaluate_survival_model(rsf_optimal, X_test, y_test)

predicted_value = []
for i in prediction:
    if i>0.4:
        predicted_value.append(1)
    else:
        predicted_value.append(0)
predicted_value = np.array(predicted_value)
performance(y_test['event'],predicted_value)


###MODEL 7 AND 8
###XGBOOST

X_train,y_train,X_test,y_test = out_of_sample_initialise()



# Assuming you have X_train, y_train, X_test, y_test as defined in the previous example

# Create XGBoost Classifier
xgb_classifier = XGBClassifier()

# Define the extended parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5],
}

# Use GridSearchCV to find the optimal number of boosting rounds
grid_search = GridSearchCV(xgb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_n_estimators = best_params['n_estimators']

print(f"Optimal number of boosting rounds: {best_n_estimators}")

# Create the XGBoost model with the optimal hyperparameters
xgb_classifier_optimal = XGBClassifier(**best_params)

# Fit the model on the entire training set
xgb_classifier_optimal.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_classifier_optimal.predict(X_test)

performance(y_test,y_pred_xgb)

print(best_params)

###LIGHT GBM

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'num_leaves': [5,10,30]
}

# Create LightGBM classifier
lgb_model = lgb.LGBMClassifier(verbose=-1,silent=True)

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(lgb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Create the model with the optimal hyperparameters
best_lgb_model = lgb.LGBMClassifier(verbose=-1,silent=True,**best_params)

# Fit the model on the entire training set
best_lgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_lgb_model.predict(X_test)

performance(y_test,y_pred)

print(best_params)

###MODEL 9 - ANN

X_train,y_train,X_test,y_test = out_of_sample_initialise()

def build_mlp_model(hidden_layers, neurons, activation):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(X_train.shape[1],)))

    for i in range(hidden_layers):
        model.add(keras.layers.Dense(neurons, activation=activation))

    model.add(keras.layers.Dense(1, activation='sigmoid'))  # Assuming binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

best_misclassification_rate = float('inf')
best_configuration = None

for hidden_layers in [1,2,3,5,7]:
    for neurons in [16,32,64]:
        for activation in ['sigmoid','relu','tanh']:
            for batch_size_given in [16,32,48]:
                 for epochs_no in [10,20,30,40]:
                    model = build_mlp_model(hidden_layers, neurons, activation)
                    model.fit(X_train, y_train, epochs=epochs_no, batch_size=batch_size_given, verbose=0)

                    # Make predictions on the validation set
                    y_pred= (model.predict(X_test) > 0.5).astype(int)

                    # Calculate misclassification rate
                    misclassification_rate = 1 - accuracy_score(y_test, y_pred)
                    roc_score = roc_auc_score(y_test,y_pred)
                    if misclassification_rate < best_misclassification_rate:
                            if roc_score >0.75:
                                best_misclassification_rate = misclassification_rate
                                y_pred_ml = y_pred
                                best_configuration = (hidden_layers, neurons, activation,batch_size_given,epochs_no)


performance(y_test,y_pred_ml)
print(best_configuration)
10.2
#filter the merge data for the test data datees
given_data = merged_data_2[(merged_data_2['YEAR']>=1991) & (merged_data_2['YEAR']<=2019)]
# Extract indices of false negatives
false_negatives_indices = (y_test == 1) & (y_pred_xgb == 0)

#find the cik and year of the companies which have been predicted false negative 
false_negatives_cik_year = pd.DataFrame(given_data.loc[false_negatives_indices, ['CIK','YEAR']])

#this dictionary will store data year-wise for each company
dict_for_10_k = dict()

#sections needed from 10-K docs
sections = ['section_1A','section_8','section_6','section_7','section_7A','section_2','section_9','section_12']

false_negatives_cik_year['CIK'] = false_negatives_cik_year['CIK'].astype("int32")

#read the json files for each cik and each year in the last five years before bankruptcy
for index,rows in false_negatives_cik_year.iterrows():
    cik = rows['CIK']
    if cik not in dict_for_10_k:
        dict_for_10_k[cik] = {}
    try:
        for year_no in range(rows['YEAR']-5,rows['YEAR']):
            dict_for_10_k[cik][year_no] = {}
            filename = f"10_K_files/{rows['CIK']}_{year_no}.json"
            # Read the JSON file
            with open(filename, 'r') as file:
                # Load the JSON content
                data = json.load(file)
                for section in sections:
                    dict_for_10_k[rows['CIK']][year_no][section] = data.get(section, "")

    except:
        pass


# Download NLTK punkt tokenizer
nltk.download('punkt')

# Iterate through the dictionary
for cik, years in dict_for_10_k.items():
    for year, sections in years.items():
        for section, content in sections.items():
            # Tokenize the content into sentences
            sentences = sent_tokenize(content)
            
            # Update the dictionary with the tokenized sentences
            dict_for_10_k[cik][year][section] = sentences

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ipuneetrathore/bert-base-cased-finetuned-finBERT")
model = AutoModelForSequenceClassification.from_pretrained("ipuneetrathore/bert-base-cased-finetuned-finBERT", from_tf=False)

# Initialize sentiment classifier
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, framework="pt")

# Mapping for sentiment labels
label_to_sentiment = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['CIK', 'Year', 'Section', 'Sentence', 'Sentiment'])

for cik, years in dict_for_10_k.items():
    for year, sections in years.items():
        for section, sentences in sections.items():
            for sentence in sentences:
                try:
                    # Analyze sentiment for each sentence
                    sentiment_result = classifier(sentence)
                    sentiment_label = sentiment_result[0]['label']
                    sentiment_category = label_to_sentiment.get(sentiment_label, 'unknown')
                    sentiment_score = sentiment_result[0]['score']
                    temp = pd.DataFrame({
                        'CIK': cik,
                        'Year': year,
                        'Section': section,
                        'sentence': sentence,
                        'Sentiment': sentiment_category,
                        'Score': sentiment_score
                    }, index = [0])
                    results_df = pd.concat([results_df,temp],axis=0)
                except:
                    pass



            
positive_result = results_df[results_df['Sentiment']=='positive']
negative_result = results_df[results_df['Sentiment']=='negative']

###DOCUMENT LEVEL MEASURE

negative_result= pd.DataFrame(negative_result.groupby(['CIK','Year'])['Sentiment'].count()).reset_index()
positive_result = pd.DataFrame(positive_result.groupby(['CIK','Year'])['Sentiment'].count()).reset_index()
total_result = pd.DataFrame(results_df.groupby(['CIK','Year'])['Sentiment'].count()).reset_index()
combined_results = pd.merge(positive_result,negative_result,on=['CIK','Year']).merge(total_result,on=['CIK','Year'])
combined_results['document_level_measure'] = (combined_results['Sentiment_x'] - combined_results['Sentiment_y'])/combined_results['Sentiment']
combined_results

###DESCRIPTIVE STATISTICS

def find_year_wise_sentiment_score(sentiment):
    # Step 1: Identify the last year for each CIK
    last_year_per_cik = results_df[results_df['Sentiment']==sentiment].groupby('CIK')['Year'].max().reset_index()

    # Step 2: Create a new column representing the difference in years
    year_desp_data = pd.merge(results_df[results_df['Sentiment']==sentiment], last_year_per_cik, on='CIK', how='left', suffixes=('', '_last_year'))
    year_desp_data['Years_Ago'] = year_desp_data['Year_last_year'] - year_desp_data['Year']

    # Optionally, you can drop the intermediate column 'Year_last_year'
    year_desp_data.drop('Year_last_year', axis=1, inplace=True)

    # Display the result
    year_desp_data = year_desp_data[(year_desp_data['Years_Ago']<=4)]
    return year_desp_data
positive_data = find_year_wise_sentiment_score('positive')
negative_data = find_year_wise_sentiment_score('negative')
neutral_data = find_year_wise_sentiment_score('neutral')


def find_descriptive_stats(data):
    # Calculate basic descriptive statistics using pandas describe
    desc_stats = data.groupby('Years_Ago')['Score'].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])

    # Calculate skewness, kurtosis, count, mean, std, max, and min for each group
    count = data.groupby('Years_Ago')['Score'].count()
    mean = data.groupby('Years_Ago')['Score'].mean()
    std = data.groupby('Years_Ago')['Score'].std()
    max_val = data.groupby('Years_Ago')['Score'].max()
    min_val = data.groupby('Years_Ago')['Score'].min()
    # Calculate skewness and kurtosis for each group
    skewness = data.groupby('Years_Ago')['Score'].apply(lambda x: x.skew())
    kurtosis = data.groupby('Years_Ago')['Score'].apply(lambda x: x.kurtosis())

    # Add additional columns to the DataFrame
    desc_stats['skewness'] = skewness
    desc_stats['kurtosis'] = kurtosis

    desc_stats['count'] = count
    desc_stats['mean'] = mean
    desc_stats['std'] = std
    desc_stats['max'] = max_val
    desc_stats['min'] = min_val

    display(desc_stats.reset_index())

    return desc_stats


print("--------POSITIVE DATA ---------------\n")
positive_desc_stats = find_descriptive_stats(positive_data)
print("--------NEGATIVE DATA ---------------\n")
negative_desc_stats = find_descriptive_stats(negative_data)
print("-------NEUTRAL DATA ---------------\n")
neutral_desc_stats = find_descriptive_stats(neutral_data)
# Group by 'Years_Ago' and calculate the mean of 'Score'
mean_scores_positive = positive_data.groupby('Years_Ago')['Score'].mean().reset_index()
mean_scores_negative = negative_data.groupby('Years_Ago')['Score'].mean().reset_index()
mean_scores_neutral = neutral_data.groupby('Years_Ago')['Score'].mean().reset_index()

# Assuming you have Series named 'mean_scores_positive', 'mean_scores_negative', 'mean_scores_neutral'
# Replace them with the actual names of your DataFrames

fig, axs = plt.subplots(3, 1, figsize=(7, 7))

# Bar Chart for Positive Scores
axs[0].bar(mean_scores_positive['Years_Ago'], mean_scores_positive['Score'], color='lightgreen', edgecolor='black', alpha=0.2)
axs[0].plot(mean_scores_positive['Years_Ago'], mean_scores_positive['Score'], color='lightgreen', linestyle='solid', linewidth=3.5, label='Mean Score')
axs[0].set_title('Bar Chart and Line for Positive Scores')
axs[0].set_xlabel('Years Ago')
axs[0].set_ylabel('Mean Score')
axs[0].set_ylim([0.986,0.99])
axs[0].invert_xaxis()

# Bar Chart for Negative Scores
axs[1].bar(mean_scores_negative['Years_Ago'], mean_scores_negative['Score'], color='salmon', edgecolor='black', alpha=0.2)
axs[1].plot(mean_scores_negative['Years_Ago'], mean_scores_negative['Score'], color='salmon', linestyle='solid', linewidth=3.5, label='Mean Score')
axs[1].set_title('Bar Chart and Line for Negative Scores')
axs[1].set_xlabel('Years Ago')
axs[1].set_ylabel('Mean Score')
axs[1].set_ylim([0.983,0.987])
axs[1].invert_xaxis()

# Bar Chart for Neutral Scores
axs[2].bar(mean_scores_neutral['Years_Ago'], mean_scores_neutral['Score'], color='skyblue', edgecolor='black', alpha=0.2)
axs[2].plot(mean_scores_neutral['Years_Ago'], mean_scores_neutral['Score'], color='skyblue', linestyle='solid', linewidth=3.5, label='Mean Score')
axs[2].set_title('Bar Chart and Line for Neutral Scores')
axs[2].set_xlabel('Years Ago')
axs[2].set_ylabel('Mean Score')
axs[2].set_ylim([0.9968, 0.9974])
axs[2].invert_xaxis()

plt.tight_layout()
plt.show()

for cols in positive_data.columns:
    positive_data[cols] = pd.to_numeric(positive_data[cols],errors='coerce')

for cols in positive_desc_stats.columns[1:]:
    print(f'\n--------------{cols}---------------\n')
    chart = alt.Chart(positive_desc_stats.reset_index()).mark_line().encode(
       x=alt.X('Years_Ago:N', title='Years Ago', sort='-x'),
        y=alt.Y(cols, title=cols, scale=alt.Scale(domain=[positive_desc_stats[cols].min(), positive_desc_stats[cols].max()])),
        color=alt.value('#90EE90')
        
    ).properties(
        width=400,
        height=200,
        title = "POSITIVE SENTIMENT"
    )
        # Create Altair chart for the shaded area
    area_chart = alt.Chart(positive_desc_stats.reset_index()).mark_area(opacity=0.3,clip=True).encode(
        x=alt.X('Years_Ago:N', title='Years Ago', sort='-x'),
        y=alt.Y(cols, title=cols),
        color=alt.value('#90EE90'),
    )

    chart_1 = alt.Chart(negative_desc_stats.reset_index()).mark_line().encode(
       x=alt.X('Years_Ago:N', title='Years Ago', sort='-x'),
        y=alt.Y(cols, title=cols, scale=alt.Scale(domain=[negative_desc_stats[cols].min(), negative_desc_stats[cols].max()])),
        color=alt.value('#FFA500')
 
    ).properties(
        width=400,
        height=200,
        title = "NEGATIVE SENTIMENT"
    )

    area_chart_1 = alt.Chart(negative_desc_stats.reset_index()).mark_area(opacity=0.3,clip=True).encode(
        x=alt.X('Years_Ago:N', title='Years Ago', sort='-x'),
        y=alt.Y(cols, title=cols),
        color=alt.value('#FFA500'),
    )
    chart_2 = alt.Chart(neutral_desc_stats.reset_index()).mark_line().encode(
       x=alt.X('Years_Ago:N', title='Years Ago', sort='-x'),
       y=alt.Y(cols, title=cols, scale=alt.Scale(domain=[neutral_desc_stats[cols].min(), neutral_desc_stats[cols].max()])),
       color=alt.value('#87CEEB')
 
    ).properties(
        width=400,
        height=200,
        title = "NEUTRAL SENTIMENT"
    )
    area_chart_2 = alt.Chart(neutral_desc_stats.reset_index()).mark_area(opacity=0.3,clip=True).encode(
        x=alt.X('Years_Ago:N', title='Years Ago', sort='-x'),
        y=alt.Y(cols, title=cols),
        color=alt.value('#87CEEB'),
    )

    display(alt.hconcat(chart+area_chart,chart_1+area_chart_1,chart_2+area_chart_2))