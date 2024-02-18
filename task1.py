# -*- coding: utf-8 -*-
"""Task1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17Mverpc8E4RR4e-6JJ04qc3DQQNJQoJG

# **Task 1**

##Data Preprocessing

###First, we import the needed libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

"""###Now assign the data set to a dataframe"""

df = pd.read_csv("first inten project.csv")

"""###Test the assignation of the dataset"""

df.head() #display the first five rows of the dataset

"""###Check the attributes (columns) of the dataset"""

df.columns

df['repeated'].value_counts()

"""###Check the general statistics of the dataset"""

df.describe()

plt.figure(figsize=(30, 10))
sns.boxplot(data=df)
plt.title('Box Plots for All Features')
plt.show()

"""###Check the data types of each attribute"""

df.info()

df['date of reservation'] = df['date of reservation'].replace('2018-2-29', '2/28/2018')

"""###Convert needed features"""

df['date of reservation'] = pd.to_datetime(df['date of reservation'])

df[df['date of reservation'].isnull()].head()

"""###Check the size of the dataset"""

df.shape

"""shows that there are 36285 rows, and 17 columns

###Check for Duplicates
"""

sum(df.duplicated(subset = "Booking_ID")) #calculates the sum of duplicate rows (if existed)

"""###Check for existing nulls"""

df.isnull().sum() #checks the sum of true null values for each column

"""results show that this dataset has 37 null values in the "date of reservation" feature

###Removing null values
"""

df = df.dropna()
df.isnull().sum() #to check is there still any null values

"""###Extract days, months and years"""

df['day'] = df['date of reservation'].dt.day.astype(int)
df['month'] = df['date of reservation'].dt.month.astype(int)
df['year'] = df['date of reservation'].dt.year.astype(int)
df.head(10)

"""###Map Season Category"""

def map_to_season(month):
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Fall'
    else:
        return 'Winter'
df['season'] = df['month'].apply(map_to_season)
df.head()

"""###Plotting Seasons


"""

season_counts = df['season'].value_counts()
seasons = season_counts.index.to_list()
counts = season_counts.to_list()
plt.bar(seasons, counts)
plt.xlabel('Seasons')
plt.ylabel('Registrations')
plt.title('Regestrations Per Season Bar Chart')

plt.show()

"""###Encode the non-numerical features"""

columns_drop = ['date of reservation', 'Booking_ID'] # dropping the date reservation, since we extracted all the info needed from
df = df.drop(columns_drop, axis = 1)
df.head()

df.select_dtypes(include = "object").head() # to view the columns only with the object type

df["booking status"] = df["booking status"].replace("Canceled", 1)
df["booking status"] = df["booking status"].replace("Not_Canceled", 0)
df["booking status"].head(20)

df["type of meal"].value_counts()

#Time to encode this feature
#Not Selected : 0, Meal Plan 1: 1,  Meal Plan 2: 2,  Meal Plan 3: 3
df["type of meal"] = df["type of meal"].replace("Not Selected", 0)
df["type of meal"] = df["type of meal"].replace("Meal Plan 1", 1)
df["type of meal"] = df["type of meal"].replace("Meal Plan 2", 2)
df["type of meal"] = df["type of meal"].replace("Meal Plan 3", 3)
df['type of meal'].head(20)

df['market segment type'].value_counts()

#We will use the one-hot encoding for this feature cause its nominal
df = pd.get_dummies(df, prefix=['market type'], columns=['market segment type'])
df.head(10)

df['room type'].value_counts()

#Time to apply the label encoding to this feature
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['room type'] = le.fit_transform(df['room type'])
df['room type'].value_counts()

#Time to encode the days and months using sine and cosine functions, sinse they are cyclical patterns
df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)

df['day_sin'] = np.sin(2*np.pi*df['day']/31)
df['day_cos'] = np.cos(2*np.pi*df['day']/31)

df.drop(['month','day'], axis = 1, inplace = True)
df.head()

df['year'].value_counts()

#Time to encode the years using label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['year'] = le.fit_transform(df['year'])
df['year'].value_counts()

df['season'].value_counts()

#Time to apply label encoding manually
df["season"] = df["season"].replace("Spring", 1)
df["season"] = df["season"].replace("Summer", 2)
df["season"] = df["season"].replace("Fall", 3)
df["season"] = df["season"].replace("Winter", 4)
df.head(10)

"""###Cleaning the numerical data"""

df_removed = df.drop(['lead time', 'average price ', 'year'], axis = 1)
plt.figure(figsize=(40, 20))
sns.boxplot(data=df_removed)
plt.title('Box Plots for All Features')
plt.show()

df["number of children"].value_counts()

# Remove rows where the number of children is more than 3
df = df[df['number of children'] <= 3]
df["number of children"].value_counts()

df["number of weekend nights"].value_counts()

df["number of week nights"].value_counts()

df["number of adults"].value_counts()

df["special requests"].value_counts()

# Remove rows where the number of adults is equal 0
df = df[df['number of adults'] > 0]
df["number of adults"].value_counts()

import matplotlib.pyplot as plt

# Assuming df is your DataFrame and 'lead_time' is the column containing lead time
plt.figure(figsize=(10, 6))
plt.hist(df['lead time'], bins=30, color='blue', edgecolor='black')
plt.title('Lead Time Distribution')
plt.xlabel('Lead Time (days)')
plt.ylabel('Registrations')
plt.grid(True)
plt.show()

# #Remove any lead time higher than 375
df = df[df['lead time'] <= 375]

#Remove now the very high average price
df = df[df['average price '] < 540]

plt.figure(figsize=(12, 6))

# Box plot for lead time
plt.subplot(1, 2, 1)
plt.boxplot(df['lead time'])
plt.title('Lead Time Box Plot')
plt.xlabel('Lead Time')

# Box plot for average price
plt.subplot(1, 2, 2)
plt.boxplot(df['average price '])
plt.title('Average Price Box Plot')
plt.xlabel('Average Price')

plt.show()

#According to my understanding to the "repeated" feature, it could be a cost feature, so it will be removed
# df.drop(['repeated'], axis = 1, inplace = True)
# df.head()

#Apply normilization to lead time and average price features
#columns_to_normalize = ["lead time", "average price "]
num_column = ['number of adults', 'number of children', 'number of weekend nights', 'number of week nights', 'lead time', 'P-C', 'P-not-C', 'average price ', 'special requests']
# Z-score scaling function
def z_score_scaling(column):
    mean_val = column.mean()
    std_dev = column.std()
    scaled_column = (column - mean_val) / std_dev
    return scaled_column


# Apply Min-Max scaling to selected columns
df[num_column] = df[num_column].apply(z_score_scaling)
df.head(10)

#Move the booking status feature to the end of the dataset
booking_status = df.pop('booking status')

df['booking status'] = booking_status
df.head()

"""###Additional Plotting"""

df[['lead time', 'average price ']].hist(bins = 20)
plt.figure(figsize=(10, 6))
plt.show()

sns.countplot(x = 'booking status', data = df)

"""###Feature Selection

"""

from sklearn.feature_selection import f_classif

X = df.drop('booking status', axis=1)
y = df['booking status']
feature_names = X.columns

# Using f_classif for feature selection
f_scores, _ =f_classif(X, y)

# Normalizing the Fisher Scores to get feature weights
feature_weights = f_scores / np.sum(f_scores)
features_df = pd.DataFrame({'Feature': feature_names, 'Fisher_Score': feature_weights})
features_df.head(30)
features_df.columns

# Plot the horizontal bar chart
features_df = features_df.sort_values(by='Fisher_Score', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(features_df['Feature'], features_df['Fisher_Score'], color='blue', alpha=0.7)
plt.xlabel('Fisher Score')
plt.ylabel('Feature')
plt.title('Fisher Scores for Each Feature')
plt.tight_layout()
plt.show()

#Time to select the top 10 features based on their scores
from sklearn.feature_selection import SelectKBest
k = 14
selected_features = features_df.head(k)['Feature'].tolist()
selected_features

#Here's now our features after selection in the dataframe
X = df[selected_features]
X.head(10)

desired_order = ['lead time', 'special requests', 'number of weekend nights', 'average price ', 'number of week nights', 'number of adults', 'year', 'month_cos', 'season', 'market type_Online', 'market type_Corporate', 'repeated', 'car parking space', 'market type_Complementary']
X = X[desired_order]
X.head()

"""###Train, Test, Spliting Sets"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

"""##Model Classification

###Logistic Regression
"""

#First we will create an object from the logistic regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=1000, random_state = 0)

#Now we will train this model
logmodel.fit(X_train, y_train)

#Let's assign the prediction from the x_test
y_pred = logmodel.predict(X_test)

#The confusion matrix here
from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#The accuracy score is here
from sklearn.metrics import accuracy_score
accuracy_log = accuracy_score(y_test, y_pred)
accuracy_log

#Finally, the classification report is here
from sklearn.metrics import classification_report
class_rep_log = classification_report(y_test, y_pred)
class_rep_log

"""###KNN"""

#First we will create an object from the knn
from sklearn.neighbors import KNeighborsClassifier
knnmod = KNeighborsClassifier(n_neighbors = 5)

#Now we will train this model
knnmod.fit(X_train, y_train)

#Let's assign the prediction from the x_test
y_pred = knnmod.predict(X_test)

#The confusion matrix here
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#The accuracy score is here
from sklearn.metrics import accuracy_score
accuracy_knn = accuracy_score(y_test, y_pred)
accuracy_knn

#Finally, the classification report is here
from sklearn.metrics import classification_report
class_rep_knn = classification_report(y_test, y_pred)
class_rep_knn

"""###SVM"""

#First we will create an object from the svc
from sklearn.svm import SVC
svcmod = SVC(kernel='linear', C=0.1)

#Now we will train this model
svcmod.fit(X_train, y_train)

#Let's assign the prediction from the x_test
y_pred = svcmod.predict(X_test)

#The confusion matrix here
from sklearn.metrics import confusion_matrix
cm_svc = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#The accuracy score is here
from sklearn.metrics import accuracy_score
accuracy_svc = accuracy_score(y_test, y_pred)
accuracy_svc

#Finally, the classification report is here
from sklearn.metrics import classification_report
class_rep_svc = classification_report(y_test, y_pred)
class_rep_svc

"""###Decision Trees"""

#First we will create an object from the decision tree
from sklearn.tree import DecisionTreeClassifier
decisionmod = DecisionTreeClassifier(random_state=1)

#Now we will train this model
decisionmod.fit(X_train, y_train)

#Let's assign the prediction from the x_test
y_pred = decisionmod.predict(X_test)

#The confusion matrix here
from sklearn.metrics import confusion_matrix
cm_decision = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_decision, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#The accuracy score is here
from sklearn.metrics import accuracy_score
accuracy_decision = accuracy_score(y_test, y_pred)
accuracy_decision

#Finally, the classification report is here
from sklearn.metrics import classification_report
class_rep_decision = classification_report(y_test, y_pred)
class_rep_decision

"""###Random Forest"""

#First we will create an object from the Random Forest
from sklearn.ensemble import RandomForestClassifier
#rfmod = RandomForestClassifier(n_estimators=50, random_state=42)

from sklearn.model_selection import GridSearchCV
#Now we will train this model
rf_model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],  # You can add more values to explore
    'max_depth': [None, 10, 20],  # You can add more values to explore
    'min_samples_split': [2, 5, 10],  # You can add more values to explore
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

#Let's assign the prediction from the x_test
y_pred = best_rf_model.predict(X_test)

#The confusion matrix here
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#The accuracy score is here
from sklearn.metrics import accuracy_score
accuracy_rf = accuracy_score(y_test, y_pred)
accuracy_rf

#Finally, the classification report is here
from sklearn.metrics import classification_report
class_rep_rf = classification_report(y_test, y_pred)
class_rep_rf

"""###Best Model Selection"""

models = ['SVC', 'Logistic Regression', 'Random Forest', 'Decision Tree', 'KNN']
scores = [accuracy_svc, accuracy_log, accuracy_rf, accuracy_decision, accuracy_knn]

plt.figure(figsize=(12, 8))
sns.barplot(x=models, y=scores, palette='Blues')
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)

# Adding error bars to represent variability
for i, score in enumerate(scores):
    plt.text(i, score + 0.0001, f'{score:.4f}', ha='center', va='bottom', color='black')

plt.show()

"""According to the above results, we see that the best algorithm is the random forest, so we can choose one of these models to work with"""

pickle.dump(best_rf_model, open('model.pkl', 'wb'))