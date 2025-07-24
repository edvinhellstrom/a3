#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


# ______________________________________________________________________
# GRADE 3

#%% Reads the datasets and prints info about them
df1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1.csv')
df2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1.csv')
df3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0.csv')

df1.info()
df2.info()
df3.info()

# %% Check the columns
print("df1 columns:", df1.columns.tolist())
print("df2 columns:", df2.columns.tolist())
print("df3 columns:", df3.columns.tolist())


# %% Check what columns that differ from each other
set(df3.columns) - set(df2.columns)


# %% Merge all the datasets into one
df = pd.concat([df1, df2, df3])

df.info()
df.head()


# %% Removes columns that is not needed and check result
dfa = df.drop(columns=['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2'])
dfa.info()
dfa.shape


#%% Replaces labels from 'normal' to 0 and other to 1
dfb = dfa
dfb['event'] = dfb['event'].replace('normal', 0)

for elem in dfb['event']:
    if elem != 0:
        dfb['event'] = dfb['event'].replace(elem, 1)

dfb['event']


# %% Splits dataset into features and labels
features = dfb.drop(columns=['event'])  
features.head()

#%% 
labels = dfb['event']                  
labels.head()


#%% Normalize the features using standardization
sc = StandardScaler() 
scaled = sc.fit_transform(features)

f_scaled = pd.DataFrame(scaled, columns=features.columns)
print(f_scaled)


# ______________________________________________________________________
# Grade 4

# %% Create a SVM classification model 
clf = svm.SVC(kernel='linear', random_state=42)


#%% Splits into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    f_scaled, labels, test_size=0.20, random_state=42)


#%% Fits the model 'clf' to the training data
clf.fit(X_train, y_train)


#%% Predicts the labels on the test data
predicted_labels = clf.predict(X_test)


#%% Evaluates the model on metrics
print(classification_report(y_test, predicted_labels, target_names=['Class 0', 'Class 1']))


# %% Perform 5-fold cross-validation on training data
clf = svm.SVC(kernel='linear', random_state=42)

scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores)


# %% Classification report on all 5 folds
y_pred = cross_val_predict(clf, X_train, y_train, cv=5)
print(classification_report(y_train, y_pred))

