# -*- coding: utf-8 -*-
"""fristTask.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/151KIkWGy3e4WE37poJxfbHntgXB0Xv4m
"""

import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pickle

#Load the dataset
df = pd.read_csv('/content/drive/MyDrive/first inten project.csv')

#Explore the data
df.head()

df.info()

df.describe().T

#Handle missing values
df['date of reservation'] = pd.to_datetime(df['date of reservation'], errors='coerce')
df["date of reservation"] = pd.to_datetime(df["date of reservation"].fillna("02/2/2018").dt.strftime("%m/%d/%Y"))

df['Month'] = df['date of reservation'].dt.month.astype('Int64')
df['Year'] = df['date of reservation'].dt.year.astype('Int64')
df['Day_of_week'] = df['date of reservation'].dt.dayofweek

df.drop(columns=['date of reservation'],axis=1,inplace=True)

df.drop(columns=['Booking_ID'],axis=1,inplace=True)

df['number of children'] = df['number of children'].fillna(0)

df.head()

# Handle noisy data
df = df[(df['number of adults'] != 0) & (df['number of children'] != 10)]
df.reset_index(drop=True, inplace=True)

# Check if the noisy data has been handled
noisy_data_handled = {
    'adults':   df[df['number of adults'] == 0],
    'children': df[df['number of children'] == 10]
}

noisy_data_handled_count = {key: len(value) for key, value in noisy_data_handled.items()}
noisy_data_handled_count

# Feature Engineering
df['total nights'] = df['number of weekend nights'] + df['number of week nights']
df['special requests ratio'] = df['special requests'] / (df['number of adults'] + df['number of children'])

df.drop(columns=['number of week nights','number of weekend nights'],inplace=True,axis=1)

#Data Visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
colors = ['#FF9900','#000000']
sns.lineplot(data=df, x='total nights', y='special requests', color=colors[1], ax=axes[0, 0])
axes[0, 0].set_title("Special Requests Over Time", size=16)
axes[0, 0].set_xlabel('total nights')
axes[0, 0].set_ylabel('Special Requests')


sns.histplot(data=df, x='average price ', kde=True, ax=axes[0, 1], color=colors[0])
axes[0, 1].set_title("Distribution of Average Price", size=16)
axes[0, 1].set_xlabel('Average Price')
axes[0, 1].set_ylabel('frequancy')


sns.boxplot(data=df, x='booking status', y='average price ', ax=axes[1, 0], color=colors[0])
axes[1, 0].set_title("Average Price  vs Cancellation Status", size=16)
axes[1, 0].set_xlabel('booking status')
axes[1, 0].set_ylabel('average price')


sns.countplot(x="booking status", data=df, ax=axes[1, 1], palette=colors)
axes[1, 1].set_title("Booking Status Count", size=16)
axes[1, 1].set_xlabel('Booking Status')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.index, y='average price ', data=df)
plt.title('Average Price Distribution')
plt.xlabel('Index')
plt.ylabel('Average Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['lead time'], bins=30, kde=True)
plt.title('Lead Time Distribution')
plt.xlabel('Lead Time (days)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df[df['booking status'] == 'Canceled']['lead time'], bins=30, kde=True, label='Canceled')
sns.histplot(df[df['booking status'] == 'Not_Canceled']['lead time'], bins=30, kde=True, label='Not Canceled')
plt.title('Impact of Lead Time on Cancellation Rates')
plt.xlabel('Lead Time (days)')
plt.ylabel('Frequency')
plt.legend(title='Booking Status')
plt.show()

booking_patterns = df.groupby('market segment type').size()
print("Booking Patterns Across Different Market Segments:")
print(booking_patterns)

cancellation_rates = df.groupby('market segment type')['booking status'].apply(lambda x: (x == 'Canceled').mean())
print("\nCancellation Rates Among Different Market Segments:")
print(cancellation_rates)

average_prices = df.groupby('market segment type')['average price '].mean()
print("\nAverage Prices Among Different Market Segments:")
print(average_prices)

plt.bar(cancellation_rates.index, cancellation_rates.values)
plt.title('Cancellation Rates Among Different Market Segments')
plt.xlabel('Market Segment Type')
plt.ylabel('Cancellation Rate')
plt.show()

plt.figure(figsize=(10, 6))
average_prices.plot(kind='bar', color='skyblue')
plt.title('Average Prices Among Different Market Segments')
plt.xlabel('Market Segment')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='room type', hue='booking status', data=df)
plt.title('Room Type vs Cancellation Likelihood')
plt.xlabel('Room Type')
plt.ylabel('Count')
plt.legend(title='Booking Status')
plt.xticks(rotation=90)
plt.show()

sns.barplot(data=df, x='room type', y='average price ')
plt.xlabel('Room Type')
plt.ylabel('Average Price')
plt.title('Average Price by Room Type')
plt.show()

# Handle outliers using IQR
def handle_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    df.loc[outliers, column] = df[column].median()  # Replace outliers with median

numerical_features = df.select_dtypes(include=[np.number]).columns
for feature in numerical_features:
    handle_outliers(df, feature)

def remove_outliers(df, threshold=3):
    cols = ['lead time', 'average price ']
    df_clean = df.copy()
            # initialize a new dataframe to avoid modifying the original
    for col in cols:
        zscore = (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
        df_clean = df_clean[abs(zscore) <= threshold]
    return df_clean

df=remove_outliers(df)

df.head()

df['booking status']=df['booking status'].map({'Not_Canceled':0,'Canceled':1})

df.head()

columns_to_encode = ['room type', 'type of meal', 'market segment type']
one_hot_encoder = OneHotEncoder(sparse=False)  # sparse=False returns a dense array
encoded_columns = one_hot_encoder.fit_transform(df[columns_to_encode])
one_hot_column_names = one_hot_encoder.get_feature_names_out(columns_to_encode)
df.drop(columns=columns_to_encode, inplace=True)
df[one_hot_column_names] = encoded_columns

df.head()

X=df.drop(columns=['booking status'])
y=df['booking status']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
sfm = SelectFromModel(clf, threshold=0.05)  # Adjust threshold as needed
sfm.fit(X, y)
selected_features = X.columns[sfm.get_support()]
print("Selected features:", selected_features)

plt.figure(figsize=(10, 8))
sns.heatmap(df[selected_features].corr(), annot=True, cmap=[colors[0], colors[1]], fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Selected Features')
plt.show()

X=df[selected_features]
y=df['booking status']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, cmap=[colors[0], colors[1]], linecolor='white', linewidth=2)
plt.title('Correlation Matrix after Outlier Handling')
plt.show()

#First of all, let's define the base DT model:

# Define the base DT model
dt_base = DecisionTreeClassifier(random_state=0)

"""In this hotel booking problem, the target variable is is_canceled, which indicates whether a booking was canceled (1) or not (0). Both false positives (a booking is predicted as canceled, but it is not) and false negatives (a booking is predicted as not canceled, but it is) could be costly for the hotel.

However, false negatives may be more costly because the hotel might lose potential customers by overbooking rooms, expecting some cancellations that do not happen. Therefore, it is important to minimize false negatives, which means maximizing recall for the 'canceled' class (1) would be a good approach.

Precision is also important because it minimizes false positives, which means minimizing the cases where the model predicts a cancellation, but the guest actually arrives. This could lead to overbooking and unsatisfied customers.

Therefore, the F1-score, which is the harmonic mean of precision and recall, would be a good metric to use as it balances both precision and recall. Specifically, the F1-score for the 'canceled' class (1) would be the most important metric for evaluating models in this project.In this hotel booking problem, the target variable is is_canceled, which indicates whether a booking was canceled (1) or not (0). Both false positives (a booking is predicted as canceled, but it is not) and false negatives (a booking is predicted as not canceled, but it is) could be costly for the hotel.

However, false negatives may be more costly because the hotel might lose potential customers by overbooking rooms, expecting some cancellations that do not happen. Therefore, it is important to minimize false negatives, which means maximizing recall for the 'canceled' class (1) would be a good approach.

Precision is also important because it minimizes false positives, which means minimizing the cases where the model predicts a cancellation, but the guest actually arrives. This could lead to overbooking and unsatisfied customers.

Therefore, the F1-score, which is the harmonic mean of precision and recall, would be a good metric to use as it balances both precision and recall. Specifically, the F1-score for the 'canceled' class (1) would be the most important metric for evaluating models in this project.v

I will create a function to identify the best set of hyperparameters that maximize the F1-score for class 1 (canceled bookings). This method provides a reusable framework for hyperparameter tuning for other models as well:
"""

def tune_clf_hyperparameters(clf, param_grid, X_train, y_train, scoring='f1', n_splits=5):
    '''
    This function optimizes the hyperparameters for a classifier by searching over a specified hyperparameter grid.
    It uses GridSearchCV and cross-validation (StratifiedKFold) to evaluate different combinations of hyperparameters.
    The combination with the highest F1-score for class 1 (canceled bookings) is selected as the default scoring metric.
    The function returns the classifier with the optimal hyperparameters.
    '''

    # Create the cross-validation object using StratifiedKFold to ensure the class distribution is the same across all the folds
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Create the GridSearchCV object
    clf_grid = GridSearchCV(clf, param_grid, cv=cv, scoring=scoring, n_jobs=-1)

    # Fit the GridSearchCV object to the training data
    clf_grid.fit(X_train, y_train)

    # Get the best hyperparameters
    best_hyperparameters = clf_grid.best_params_

    # Return best_estimator_ attribute which gives us the best model that has been fitted to the training data
    return clf_grid.best_estimator_, best_hyperparameters

# Hyperparameter grid for DT
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [13, 14, 15],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'class_weight': [{0: 1, 1: w} for w in [1, 2, 3]]
    #Since the data is slightly imbalanced and we want to optimize for class 1,
    #we have included the class_weight parameter in our grid.
    #In the grid above, the weight for class 0 is always 1,
    #while the weight for class 1 varies from 1 to 5.
    #This will help the model to focus more on class 1.
}

# Call the function for hyperparameter tuning
best_dt, best_dt_hyperparams = tune_clf_hyperparameters(dt_base, param_grid_dt, x_train, y_train)

print('DT Optimal Hyperparameters: \n', best_dt_hyperparams)

def metrics_calculator(clf, X_test, y_test, model_name):
    '''
    This function calculates all desired performance metrics for a given model on test data.
    The metrics are calculated specifically for class 1.
    '''
    y_pred = clf.predict(X_test)
    result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, pos_label=1),
                                recall_score(y_test, y_pred, pos_label=1),
                                f1_score(y_test, y_pred, pos_label=1),
                                roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])],
                          index=['Accuracy','Precision (Class 1)','Recall (Class 1)','F1-score (Class 1)','AUC (Class 1)'],
                          columns = [model_name])

    result = (result * 100).round(2).astype(str) + '%'
    return result

def model_evaluation(clf, X_train, X_test, y_train, y_test, model_name):
    '''
    This function provides a complete report of the model's performance including classification reports,
    confusion matrix and ROC curve.
    '''
    sns.set(font_scale=1.2)

    # Generate classification report for training set
    y_pred_train = clf.predict(X_train)
    print("\n\t  Classification report for training set")
    print("-"*55)
    print(classification_report(y_train, y_pred_train))

    # Generate classification report for test set
    y_pred_test = clf.predict(X_test)
    print("\n\t   Classification report for test set")
    print("-"*55)
    print(classification_report(y_test, y_pred_test))

    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=100, gridspec_kw={'width_ratios': [2, 2, 1]})

    # Define a colormap
    royalblue = LinearSegmentedColormap.from_list('royalblue', [(0, (1,1,1)), (1, (0.25,0.41,0.88))])
    royalblue_r = royalblue.reversed()

    # Plot confusion matrix for test set
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, colorbar=False, cmap=royalblue, ax=ax1)
    ax1.set_title('Confusion Matrix for Test Data')
    ax1.grid(False)

    # Plot ROC curve for test data and display AUC score
    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax2)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve for Test Data (Positive label: 1)')

    # Report results for the class specified by positive label
    result = metrics_calculator(clf, X_test, y_test, model_name)
    table = ax3.table(cellText=result.values, colLabels=result.columns, rowLabels=result.index, loc='center')
    table.scale(0.6, 2)
    table.set_fontsize(12)
    ax3.axis('tight')
    ax3.axis('off')
    # Modify color
    for key, cell in table.get_celld().items():
       if key[0] == 0:
           cell.set_color('royalblue')
    plt.tight_layout()
    plt.show()

model_evaluation(best_dt, x_train, x_test, y_train, y_test, 'Decision Tree')

"""Training Set Classification Report:
Precision: Precision measures the proportion of true positive predictions among all instances predicted as positive. In this case, the precision for class 0 (not canceled) is 0.93, indicating that 93% of instances predicted as not canceled were indeed not canceled. For class 1 (canceled), the precision is 0.76, indicating that 76% of instances predicted as canceled were indeed canceled.

Recall: Recall (also known as sensitivity or true positive rate) measures the proportion of actual positive instances that were correctly identified by the model. For class 0, the recall is 0.87, indicating that 87% of actual not canceled instances were correctly classified. For class 1, the recall is 0.87, indicating that 87% of actual canceled instances were correctly classified.

F1-score: The F1-score is the harmonic mean of precision and recall, providing a balanced measure of a model's performance. It considers both false positives and false negatives. The weighted average F1-score for class 0 is 0.90, and for class 1, it is 0.81.
Support: Support refers to the number of actual occurrences of each class in the dataset.

Test Set Classification Report:
The classification report for the test set presents similar metrics as the training set, but it evaluates the model's performance on unseen data.
The precision, recall, and F1-score for both classes (0 and 1) are slightly lower compared to the training set. This drop in performance is expected as the model encounters new data during testing.
The accuracy of the model on the test set is 0.78, indicating that it correctly predicts the class for approximately 78% of the instances in the test set.

The confusion matrix shows that there are still some False Positives and False Negatives, but the model is doing a relatively good job of minimizing them.

Summary:
Overall, the classification reports provide a detailed overview of the model's performance in terms of precision, recall, and F1-score for each class, along with the overall accuracy. They help in understanding how well the model generalizes to unseen data and identifies areas for potential improvement.
"""

# Save the final performance of DT classifier
dt_result = metrics_calculator(best_dt, x_test, y_test, 'Decision Tree')
dt_result

# Define the base RF model
rf_base = RandomForestClassifier(random_state=0, n_jobs=-1)

param_grid_rf = {
    'n_estimators': [100, 50],
    'criterion': ['entropy'],
    'max_depth': [16, 18],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2],
    'class_weight': [{0: 1, 1: w} for w in [1, 2]]
}

# Using the tune_clf_hyperparameters function to get the best estimator
best_rf, best_rf_hyperparams = tune_clf_hyperparameters(rf_base, param_grid_rf, x_train, y_train)

print('RF Optimal Hyperparameters: \n', best_rf_hyperparams)

model_evaluation(best_rf, x_train, x_test, y_train, y_test, 'Random Forest')

"""Training Set:
Accuracy: The model correctly predicts whether a booking will be canceled or not about 97% of the time.

Precision (Class 1): Approximately 96% of the bookings that the model predicted as canceled were actually canceled.

Recall (Class 1): The model correctly identified approximately 96% of the actual cancellations.

F1-score (Class 1): The harmonic mean of precision and recall for class 1 is 96%.

Macro Avg: The average of precision, recall, and F1-score across both classes.

Weighted Avg: The weighted average of precision, recall, and F1-score, weighted by the support (number of true instances) for each class.

Test Set:
Accuracy: The model correctly predicts whether a booking will be canceled or not about 83% of the time on unseen data.

Precision (Class 1): Approximately 78% of the bookings that the model predicted as canceled were actually canceled.

Recall (Class 1): The model correctly identified approximately 68% of the actual cancellations.

F1-score (Class 1): The harmonic mean of precision and recall for class 1 is 73%.

Macro Avg: The average of precision, recall, and F1-score across both classes.

Weighted Avg: The weighted average of precision, recall, and F1-score, weighted by the support (number of true instances) for each class.

Compared to the Decision Tree (DT) model, the Random Forest (RF) model shows a significant improvement in all the performance metrics, particularly in reducing the number of False Positives. This indicates that the Random Forest model is better at balancing the trade-off between Precision and Recall, leading to a higher F1-score.

Overall:
The model performs well on the training set, achieving high accuracy, precision, recall, and F1-score for both classes.
However, there is a drop in performance on the test set, indicating some degree of overfitting or lack of generalization to unseen data.
The model shows better performance in predicting non-canceled bookings (class 0) compared to canceled bookings (class 1) on the test set, as indicated by the higher precision, recall, and F1-score for class 0.
"""

rf_result = metrics_calculator(best_rf, x_test, y_test, 'Random Forest')
rf_result

"""XGBoost, which stands for eXtreme Gradient Boosting, is an efficient and scalable implementation of gradient boosting. It is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. It is particularly popular and effective for classification and regression tasks and has gained reputation for its performance and speed compared to other implementations of gradient boosting. XGBoost is designed to be efficient, flexible and portable.

First of all, let's define the base XGBoost model. We set use_label_encoder=False to avoid a deprecation warning. The eval_metric is set to 'logloss' because it is a more appropriate metric for binary classification tasks and for the slight class imbalance we have in our target variable:
"""

# Define the model
xgb_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)

"""XGBoost has several hyperparameters that need to be tuned to improve the performance of the model. Some of the important hyperparameters are:

n_estimators: This is the number of boosting rounds or trees to be built. It is usually set to a high number, but XGBoost has an early stopping feature that stops the model building process when no further improvements are observed.
learning_rate: This is the step size shrinkage used to prevent overfitting. It ranges from 0 to 1.

max_depth: This is the maximum depth of a tree and can range from 1 to infinity.

subsample: This is the fraction of observations to be randomly sampled for each tree. It ranges from 0 to 1.

colsample_bytree: This is the fraction of features to be randomly sampled for each tree. It ranges from 0 to 1.
"""

param_grid_xgb = {
    'n_estimators': [250, 350],
    'learning_rate': [0.01, 0.1],
    'max_depth': [7, 8],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.8, 0.9],
    'scale_pos_weight': [1, (y_train == 0).sum() / (y_train == 1).sum()]
}

# Call the function for hyperparameter tuning
best_xgb, best_xgb_hyperparams = tune_clf_hyperparameters(xgb_base, param_grid_xgb, x_train, y_train)

print('XGBoost Optimal Hyperparameters: \n', best_xgb_hyperparams)

model_evaluation(best_xgb, x_train, x_test, y_train, y_test, 'XGBoost')

"""Training Set Classification Report:
Precision: Precision measures the proportion of true positive predictions among all instances predicted as positive. In this report, the precision for class 0 (not canceled) is 0.97, indicating that 97% of instances predicted as not canceled were indeed not canceled. For class 1 (canceled), the precision is 0.87, indicating that 87% of instances predicted as canceled were indeed canceled.

Recall: Recall (also known as sensitivity or true positive rate) measures the proportion of actual positive instances that were correctly identified by the model. For class 0, the recall is 0.93, indicating that 93% of actual not canceled instances were correctly classified. For class 1, the recall is 0.94, indicating that 94% of actual canceled instances were correctly classified.

F1-score: The F1-score is the harmonic mean of precision and recall, providing a balanced measure of a model's performance. It considers both false positives and false negatives. The weighted average F1-score for class 0 is 0.95, and for class 1, it is 0.90.

Accuracy: Accuracy measures the overall correctness of the model's predictions. In this report, the accuracy on the training set is 0.93, indicating that the model correctly predicts the class for approximately 93% of the instances in the training set.
Test Set Classification Report:
The classification report for the test set presents similar metrics as the training set but evaluates the model's performance on unseen data.

The precision, recall, and F1-score for both classes (0 and 1) are slightly lower compared to the training set. This drop in performance is expected as the model encounters new data during testing.
The accuracy of the model on the test set is 0.82, indicating that it correctly predicts the class for approximately 82% of the instances in the test set.
"""

xgb_result = metrics_calculator(best_xgb, x_test, y_test, 'XGBoost')
xgb_result

from sklearn.neighbors import KNeighborsClassifier

# Define the base KNN model
knn_base = KNeighborsClassifier()

# Hyperparameter grid for KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree']
}

best_knn, best_knn_hyperparams = tune_clf_hyperparameters(knn_base, param_grid_knn, x_train, y_train)
print('KNN Optimal Hyperparameters: \n', best_knn_hyperparams)

model_evaluation(best_knn, x_train, x_test, y_train, y_test, 'KNN')

knn_result = metrics_calculator(best_knn, x_test, y_test, 'KNN')
knn_result

from sklearn.ensemble import GradientBoostingClassifier

# Define the base GBM model
gbm_base = GradientBoostingClassifier(random_state=0)

param_grid_gbm = {
    'n_estimators': [100, 150],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4],
    'subsample': [0.8, 1.0]
}

# Call the function for hyperparameter tuning
best_gbm, best_gbm_hyperparams = tune_clf_hyperparameters(gbm_base, param_grid_gbm, x_train, y_train)
print('GBM Optimal Hyperparameters: \n', best_gbm_hyperparams)

model_evaluation(best_gbm, x_train, x_test, y_train, y_test, 'GBM')

gbm_result = metrics_calculator(best_gbm, x_test, y_test, 'GBM')
gbm_result

from sklearn.naive_bayes import GaussianNB

# Define the base Naive Bayes model
nb_base = GaussianNB()

# Hyperparameter grid for Naive Bayes
param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7] #These values represent different levels of smoothing applied to the variance.
                                         #Smaller values indicate less smoothing
                                         #var_smoothing can help you find the best balance between stability and sensitivity in your model.
}

best_nb, best_nb_hyperparams = tune_clf_hyperparameters(nb_base, param_grid_nb,x_train,y_train)
print('NB Optimal Hyperparameters: \n', best_nb_hyperparams)

model_evaluation(best_nb, x_train, x_test, y_train, y_test, 'NB')

nb_result = metrics_calculator(best_nb, x_test, y_test, 'NB')
nb_result

results = pd.concat([dt_result, rf_result, xgb_result,knn_result,gbm_result,nb_result], axis=1).T

# Sort the dataframe in descending order based on F1-score (class 1) values
results.sort_values(by='F1-score (Class 1)', ascending=False, inplace=True)

# Color the F1-score column
results.style.applymap(lambda x: 'background-color: royalblue', subset='F1-score (Class 1)')

with open('best_xgb.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)