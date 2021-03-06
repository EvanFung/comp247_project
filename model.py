# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:31:10 2022

@author: Evan FENG
"""
import pandas as pd, numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('KSI.csv')
#printing percentage of missing values for each feature
data.replace('<Null>', np.nan, inplace=True)
data.replace('None', np.nan, inplace=True)
data.replace(' ',np.nan,inplace=True)
print(data.isna().sum()/len(data)*100)
#The following heatmap shows the features having maximum missing values
fig, ax = plt.subplots(figsize=(15,7))
#heatmap to visualize features with most missing values
sns.heatmap(data.isnull(), yticklabels=False,cmap='Greens')
#shape
print(data.shape)
# Dropping columns where missing values were greater than 80%
drop_column = ['OFFSET','FATAL_NO','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND','ObjectId','X','Y','INDEX_']
data.drop(drop_column, axis=1, inplace=True)
print(data.shape)
print(data.isna().sum()/len(data)*100)

print(data.info())
print(data.describe())

#Changing the property damage and non-fatal columns to Non-Fatal¶
data['ACCLASS'] = np.where(data['ACCLASS'] == 'Property Damage Only', 'Non-Fatal', data['ACCLASS'])
data['ACCLASS'] = np.where(data['ACCLASS'] == 'Non-Fatal Injury', 'Non-Fatal', data['ACCLASS'])
data['ACCLASS'].unique()


## Verifying columns with object data type
print(data.select_dtypes(["object"]).columns)

# Neighbourhood is identical with Hood ID
data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True)
# change data type
data['DATE'] = pd.to_datetime(data['DATE'])
data['DAY'] = pd.to_datetime(data['DATE']).dt.day
data['MONTH'] = data['DATE'].dt.month
data['MINUTE'],data['SECOND'] = divmod(data['TIME'], 60)
df_timestamp = pd.DataFrame({'year':data['YEAR'],'month':data['MONTH'],'day':data['DAY'],'hour':data['HOUR'],'minute':data['MINUTE'],'second':data['SECOND']})
data['TIMESTAMP'] = pd.to_datetime(df_timestamp)

#Number of Unique accidents by Year
Num_accident = data.groupby('YEAR')['ACCNUM'].nunique()
plt.figure(figsize=(12,6))
plt.title("Accidents caused in different years")
plt.ylabel('Number of Accidents (ACCNUM)')
ax = plt.gca()
ax.tick_params(axis='x', colors='blue')
ax.tick_params(axis='y', colors='red')
my_colors = list('rgbkymc')   #red, green, blue, black, etc.
Num_accident.plot(
    kind='bar', 
    color='blue',
    edgecolor='black'
)
#Num_accident.plot(kind='bar',color= my_colors)
plt.show()

#Number of Unique accidents by Month
Num_accident = data.groupby('MONTH')['ACCNUM'].nunique()
plt.figure(figsize=(12,6))
plt.title("Accidents caused in different months")
plt.ylabel('Number of Accidents (ACCNUM)')



ax = plt.gca()
ax.tick_params(axis='x', colors='blue')
ax.tick_params(axis='y', colors='red')
my_colors = list('rgbkymc')   #red, green, blue, black, etc.
Num_accident.plot(
    kind='bar', 
    color='blue',
    edgecolor='black'
)
#Num_accident.plot(kind='bar',color= my_colors)
plt.show()

#From the data above, accidents happened more from June to October

#Categorizing Fatal vs. non-Fatal Incident (non-unique i.e: one accident is counted depending upon involved parties)
plt.xticks(rotation=70)
plt.tight_layout()
sns.catplot(x='YEAR', kind='count', data=data,  hue='ACCLASS')
plt.xticks(rotation=0)
plt.tight_layout()


#Looking at area where accident happens
Region_data = data['DISTRICT'].value_counts()
plt.figure(figsize=(12,6))
plt.ylabel('Number of Accidents')
Region_data.plot(kind='bar',color=list('rgbkmc'))
plt.show()

#data cleaning
data.shape
data.columns
data.dtypes

clean_data = data[['ACCNUM','YEAR','HOUR','MONTH','DAY','MINUTE','SECOND',
                   'DISTRICT','LATITUDE','LONGITUDE','HOOD_ID','VISIBILITY',
                   'LIGHT','RDSFCOND','PEDESTRIAN','CYCLIST','AUTOMOBILE',
                   'MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','PASSENGER',
                   'SPEEDING','AG_DRIV','REDLIGHT','ALCOHOL','DISABILITY',
                   ]]

#Driving condition for accidents
#AG_DRIV
#REDLIGHT
#ALCOHOL
#DISABILITY
#SPEEDING

#Type of vehicles involved
#AUTOMOBILE
#MOTORCYCLE
#TRUCK
#CYCLIST
#EMERG_VEH
#TRNS_CITY_VEH

#Type of victims
#CYCLIST
#PEDESTRIAN
#PASSENGER

#We noted that if the value of these columns are nan which means No
#Therefore we need to transform yes to 1, nan to 0

#Changing the nan to 0 and Yes to 1 for alcohol
clean_data['ALCOHOL'] = np.where(clean_data['ALCOHOL'] == 'Yes',1,0)
clean_data['AG_DRIV'] = np.where(clean_data['AG_DRIV'] == 'Yes',1,0)
clean_data['REDLIGHT'] = np.where(clean_data['REDLIGHT'] == 'Yes',1,0)
clean_data['DISABILITY'] = np.where(clean_data['DISABILITY'] == 'Yes',1,0)
clean_data['SPEEDING'] = np.where(clean_data['SPEEDING'] == 'Yes',1,0)

clean_data['AUTOMOBILE'] = np.where(clean_data['AUTOMOBILE'] == 'Yes',1,0)
clean_data['MOTORCYCLE'] = np.where(clean_data['MOTORCYCLE'] == 'Yes',1,0)
clean_data['TRUCK'] = np.where(clean_data['TRUCK'] == 'Yes',1,0)
clean_data['CYCLIST'] = np.where(clean_data['CYCLIST'] == 'Yes',1,0)
clean_data['EMERG_VEH'] = np.where(clean_data['EMERG_VEH'] == 'Yes',1,0)
clean_data['TRSN_CITY_VEH'] = np.where(clean_data['TRSN_CITY_VEH'] == 'Yes',1,0)

clean_data['PEDESTRIAN'] = np.where(clean_data['PEDESTRIAN'] == 'Yes',1,0)
clean_data['PASSENGER'] = np.where(clean_data['PASSENGER'] == 'Yes',1,0)

# Neighbourhood is identical with Hood ID so we drop Neighbourhood
# duplicated or not related data , we drop.
# after drop data
clean_data.shape
clean_data.columns
clean_data.dtypes
print(clean_data.isna().sum()/len(clean_data)*100)
#there are 3 columns which has null value
clean_data['DISTRICT'].unique()
clean_data['VISIBILITY'].unique()
clean_data['RDSFCOND'].unique()
#141
clean_data['DISTRICT'].isna().sum()
#18
clean_data['VISIBILITY'].isna().sum()
#23
clean_data['RDSFCOND'].isna().sum()
#For district,visibility,rdsfcon, fill nan with Other
#clean_data['DISTRICT'] = clean_data['DISTRICT'].fillna('Other')
#clean_data['VISIBILITY'] = clean_data['VISIBILITY'].fillna('Other')
#clean_data['RDSFCOND'] = clean_data['RDSFCOND'].fillna('Other')

objdtype_cols = clean_data.select_dtypes(["object"]).columns
clean_data[objdtype_cols] = clean_data[objdtype_cols].astype('category')

clean_data['LATITUDE']=clean_data['LATITUDE'].astype('int')
clean_data['LONGITUDE']=clean_data['LATITUDE'].astype('int')


clean_data_target = data['ACCLASS']
clean_data_target = np.where(clean_data_target == 'Fatal',1,0)

clean_data.columns

clean_data = pd.get_dummies(clean_data, columns=['VISIBILITY','RDSFCOND','DISTRICT','LIGHT'])
scaler = StandardScaler() #define the instance
scaler.fit_transform(clean_data)


#Feature selection
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X_1 = sm.add_constant(clean_data)

#Fitting sm.OLS model
model = sm.OLS(clean_data_target,X_1).fit()
model.pvalues
model.pvalues>0.1

X_droped_0 = clean_data.drop(['ACCNUM','MONTH','DAY','LATITUDE','LONGITUDE',
                              'SECOND','HOOD_ID','MOTORCYCLE','CYCLIST',
                              'AUTOMOBILE','EMERG_VEH','ALCOHOL','VISIBILITY_Clear',
                              'VISIBILITY_Fog, Mist, Smoke, Dust','VISIBILITY_Freezing Rain',
                              'VISIBILITY_Other','VISIBILITY_Rain','VISIBILITY_Snow',
                              'VISIBILITY_Strong wind','LIGHT_Dark','LIGHT_Dark, artificial',
                              'LIGHT_Dawn','LIGHT_Dawn, artificial','LIGHT_Daylight, artificial',
                              'LIGHT_Dusk','LIGHT_Other'],axis=1)

X_droped_0.shape
print(X_droped_0.isna().sum()/len(X_droped_0)*100)


#Model for prediction
y = clean_data_target
X = X_droped_0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

X.columns

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)
print("[Logistic regression algorithm] accuracy_score: {:.3f}.".format(acc))
print('metric on test set\n', classification_report(y_test, y_pred))

#To do
#imbalanced classes handling
#pipelines
#grid search

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from time import time

# Modeling Decision Trees
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=3, criterion = 'entropy', random_state=42)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Validate using 10 cross fold

clf = DecisionTreeClassifier(min_samples_split=20,criterion = 'entropy',
                                random_state=42)
clf.fit(X_train, y_train)
scores= cross_val_score(\
   clf, X_train, y_train, cv=10, scoring='f1_macro')

print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                          scores.std()),
                                         end="\n\n" )
#Predict using the test set
y_pred = clf.predict(X_test)
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

"""
Tunning the model

"""
# set of parameters to test
param_grid = {
              "min_samples_split": [2, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }
print("-- Grid Parameter Search via 10-fold CV")
dt = DecisionTreeClassifier(criterion = 'entropy')
grid_search = GridSearchCV(dt,
                               param_grid=param_grid,
                               cv=10)
start = time()
grid_search.fit(X_train, y_train)

print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.cv_results_)))

print('Best Parameters are:',grid_search.best_params_)

#Predict the response for test dataset using the best parameters
dt = DecisionTreeClassifier(max_depth =5,min_samples_split= 2, criterion = 'entropy', min_samples_leaf= 10, random_state=42 )
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


