# -*- coding: utf-8 -*-
"""247project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/172YMf0i-JUFCPXNyzO-G-QuLg3wpmapn
"""

import pandas as pd, numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score, precision_score, recall_score

data = pd.read_csv('KSI.csv')
print("Initial shape:", data.shape)

# There are several columns consist of "Yes" and "<Null>" (where Null means No). 
# For these binary column, replace  "<Null>" with"No"
binary_cols = ['CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','REDLIGHT','ALCOHOL','DISABILITY','PASSENGER','AG_DRIV','PEDESTRIAN']
data[binary_cols]=data[binary_cols].replace({'<Null>':'No', 'Yes':'Yes'})

# Replace other '<Null>' with nan, printing percentage of missing values for each feature
data.replace('<Null>', np.nan, inplace=True)
data.replace(' ',np.nan,inplace=True)
print(data.isna().sum()/len(data)*100)

#The following heatmap shows the features having maximum missing values
fig, ax = plt.subplots(figsize=(15,7))
#heatmap to visualize features with most missing values
sns.heatmap(data.isnull(), yticklabels=False,cmap='Greens')
#shape
print(data.shape)

# Dropping columns where missing values were greater than 80%
drop_column = ['OFFSET','FATAL_NO','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND']
data.drop(drop_column, axis=1, inplace=True)
#Drop irrelevant columns which are unique identifier
data.drop(['ObjectId','INDEX_'], axis=1, inplace=True)

print(data.shape)
print(data.isna().sum()/len(data)*100)

print(data.info())


#Changing the property damage and non-fatal columns to Non-Fatal¶
data['ACCLASS'] = np.where(data['ACCLASS'] == 'Property Damage Only', 'Non-Fatal', data['ACCLASS'])
data['ACCLASS'] = np.where(data['ACCLASS'] == 'Non-Fatal Injury', 'Non-Fatal', data['ACCLASS'])

data['ACCLASS'].unique()

## Verifying columns with object data type
print(data.select_dtypes(["object"]).columns)


# Neighbourhood is identical with Hood ID
#data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True) # Neighbourhood is identical with Hood ID

# extract features: weekday,day, month 
data['DATE'] = pd.to_datetime(data['DATE'])
data['WEEKDAY'] =data['DATE'].dt.dayofweek
data['DAY'] = pd.to_datetime(data['DATE']).dt.day
data['MONTH'] = data['DATE'].dt.month

#Drop Date
data.drop(['DATE'], axis=1, inplace=True)

# Neighbourhood is identical with Hood ID, drop Neighbourhood
# X,Y are longitude and latitudes, dulicate, drop X and Y
data.drop(['NEIGHBOURHOOD','X','Y'], axis=1, inplace=True)

data['STREET1'].value_counts()
data['POLICE_DIVISION'].value_counts() 
# remove other irrelevant columns or columns contain too many missing values
data.drop(['MANOEUVER','DRIVACT','DRIVCOND','INITDIR','STREET1','STREET2','WARDNUM','POLICE_DIVISION','DIVISION'], axis=1, inplace=True)

#Injury
ax=sns.catplot(x='INJURY', kind='count', data=data,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("INJURY")

data['INJURY'].value_counts()

# Injury too closely related to fatal/non-fatal, drop
data.drop(['INJURY'], axis=1, inplace=True)

"""#Visualization"""

#Number of Unique accidents by Year
Num_accident = data.groupby('YEAR')['ACCNUM'].nunique()
plt.figure(figsize=(12,6))
plt.title("Accidents in different years")
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
plt.title("Accidents in different months")
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

#Number of Unique accidents by Day
Num_accident = data.groupby('WEEKDAY')['ACCNUM'].nunique()
plt.figure(figsize=(12,6))
plt.title("Accidents in different weekdays")
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

#Check the relation between features and target
#Year
ax=sns.catplot(x='YEAR', kind='count', data=data,  hue='ACCLASS')
ax.set_xticklabels(rotation=45)
plt.title("Accidents in different years")


#Month
ax=sns.catplot(x='MONTH', kind='count', data=data,  hue='ACCLASS')
plt.title("Accidents in different months")


#Month
ax=sns.catplot(x='WEEKDAY', kind='count', data=data,  hue='ACCLASS')
plt.title("Accidents in different day of a week")

#Neighborhood
ax=sns.catplot(x='DISTRICT', kind='count', data=data,  hue='ACCLASS')
ax.set_xticklabels(rotation=45)
plt.title("Accidents in different day of a week")

#Vehicle type
ax=sns.catplot(x='VEHTYPE', kind='count', data=data,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Vehicle type vs. occurance of accidents")

#LOCCOORD
ax=sns.catplot(x='LOCCOORD', kind='count', data=data,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Location Coordinate")


#RDSFCOND
ax=sns.catplot(x='RDSFCOND', kind='count', data=data,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Road Surface Condition")


#INVAGE
ax=sns.catplot(x='INVAGE', kind='count', data=data,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Age of Involved Party")


#Light
ax=sns.catplot(x='LIGHT', kind='count', data=data,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Light condition")

# Where does accident more likly to occur
#2D histogram
plt.hist2d(data['LATITUDE'], data['LONGITUDE'], bins=(40, 40), cmap=plt.cm.jet)
plt.title("2D histogram of all fatal and non-fatal accidents")
plt.xlabel("LATITUDE")
plt.ylabel("LONGITUDE")
plt.show()

data_Fatal = data[data['ACCLASS'] == 'Fatal']
plt.hist2d(data_Fatal['LATITUDE'], data_Fatal['LONGITUDE'], bins=(40, 40), cmap=plt.cm.jet)
plt.title("2D histogram of fatal accidents")
plt.xlabel("LATITUDE")
plt.ylabel("LONGITUDE")
plt.show()


# scatter plot of all fatal and non-fatal accidents
sns.scatterplot(x='LATITUDE', y='LONGITUDE', data = data, hue = "ACCLASS",alpha=0.3)
plt.title("Accidents")
plt.show()
#scatter plot of fatal accidents
sns.scatterplot(x='LATITUDE', y='LONGITUDE', data = data[data['ACCLASS'] == 'Fatal'],alpha=0.3)
plt.title("Fatal Accidents")
plt.show()

"""#Further Data Cleaning"""

print(data.isna().sum()/len(data)*100)

#several columns <3% missing values, 
#catagorical feature, not make much sense if impute, so keep the features, just discard these rows with missing values
data.dropna(subset=['ROAD_CLASS', 'DISTRICT','VISIBILITY','RDSFCOND','LOCCOORD','IMPACTYPE','TRAFFCTL','INVTYPE'],inplace=True)

#target class
data['ACCLASS']=data['ACCLASS'].replace({'Non-Fatal':0, 'Fatal':1})
data['ACCLASS'].value_counts()   #dataset is unbalanced

#Resampling- Upsampled

from sklearn.utils import resample
df=data
df_majority = df[df.ACCLASS==0]
df_minority = df[df.ACCLASS==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=14029,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
print(df_upsampled.ACCLASS.value_counts())

data=df_upsampled

#Test Train split
#Since the dataset is unbalanced, use straified split
X = data.drop(["ACCLASS"], axis=1)
y= data["ACCLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5,stratify=y)

#impute
from sklearn.impute import SimpleImputer    
imputer = SimpleImputer(strategy="constant",fill_value='missing')  
data_tr=imputer.fit_transform(X_train)
data_tr= pd.DataFrame(data_tr, columns=X_train.columns)

print(data_tr.isna().sum()/len(data_tr)*100)

#numerical features
df1=data.drop(['ACCLASS'],axis=1)
num_columns=df1.select_dtypes(include=[np.number]).columns
print(num_columns)
data_num =data_tr[num_columns] 
#standardize 
scaler = StandardScaler() #define the instance
scaled =scaler.fit_transform(data_num)
data_num_scaled= pd.DataFrame(scaled, columns=num_columns)

#categorical features
cat_columns=df1.select_dtypes(exclude=[np.number]).columns
print(cat_columns)
categoricalData =data_tr[cat_columns]

data_cat = pd.get_dummies(categoricalData, columns=cat_columns, drop_first=True)
data_cat

X_train_prepared=pd.concat([data_num_scaled, data_cat], axis=1)
X_train_prepared

"""#Feature Selection"""

#method 1: using SelectFromModel and RandomForestClassifier to select features
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train_prepared, y_train)
selected_feat= X_train_prepared.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)

feat_importances = pd.Series(sel.estimator_.feature_importances_, index=X_train_prepared.columns)
#plot the selected features
fig = plt.gcf()
fig.set_size_inches(12, 8)
feat_importances.nlargest(30).plot(kind='barh')
plt.title("Feature Importance")
plt.show()

#method 3: Logistic regression
from sklearn.linear_model import LogisticRegression
sel = SelectFromModel(LogisticRegression(solver='saga',penalty='l1'))
sel.fit(X_train_prepared, y_train)
selected_feat= X_train_prepared.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)

coefficient= pd.Series(sel.estimator_.coef_[0], index=X_train_prepared.columns)
#plot the selected features
fig = plt.gcf()
fig.set_size_inches(12, 30)
coefficient.plot(kind='barh')
plt.title("L1 coefficient")
plt.show()

abs_coefficient =abs(coefficient)
print(coefficient[coefficient==0])

#selected features

#numerical features
num_columns=['ACCNUM', 'YEAR', 'TIME', 'HOUR', 'LATITUDE', 'LONGITUDE', 'WEEKDAY', 'DAY', 'MONTH']
data_num =data_tr[num_columns] 
num_columns=data_num.columns
print(num_columns)

#categorical features

cat_columns=['CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','REDLIGHT','ALCOHOL','DISABILITY','PASSENGER','AG_DRIV','PEDESTRIAN',
              'ROAD_CLASS', 'DISTRICT',  'TRAFFCTL','VISIBILITY', 'LIGHT', 'RDSFCOND','IMPACTYPE', 'INVAGE']
categoricalData =data_tr[cat_columns]
#categoricalData= data_tr.select_dtypes(exclude=[np.number])
print(categoricalData.columns)
data_cat = pd.get_dummies(categoricalData, columns=cat_columns, drop_first=True)
data_cat

df=pd.concat([data_num, data_cat], axis=1)
df

##################

# Pipelines

#################
# build a pipeline for preprocessing the categorical attributes
cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant",fill_value='missing')),
        ('one_hot', OneHotEncoder(drop='first')),
    ])
# build a pipeline for preprocessing the numerical attributes
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
#full transformation Column Transformer
num_attribs = num_columns
cat_attribs = cat_columns

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

X_train_prepared = full_pipeline.fit_transform(X_train)
X_train_prepared.toarray()

"""# Model Training,Tuning and Testing

##Logistic regression

##SVM

Before Tuning
"""

X_test_prepared = full_pipeline.transform(X_test)
X_train_prepared.shape

#SVM
from sklearn.svm import SVC
clf=SVC()
X_train_prepared = full_pipeline.fit_transform(X_train)
clf.fit(X_train_prepared, y_train)
#accuracy on training dataset
print("Training Accuracy",clf.score(X_train_prepared,y_train))

#test
X_test_prepared = full_pipeline.transform(X_test)
#predict
y_test_pred=clf.predict(X_test_prepared)

print("Before Tuning:")
print("accuracy", accuracy_score(y_test, y_test_pred))
print("precison",precision_score(y_test, y_test_pred))
print("recall",recall_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

"""After Tuning"""

#Fine Tuning
X_train_prepared = full_pipeline.fit_transform(X_train)

#SVM
from sklearn.svm import SVC
clf_svm=SVC()

#Random search
param_svm = [
    {'kernel': ['linear', 'poly','rbf'], 
     'C': [0.01,0.1, 1],
     'gamma': [0.01, 0.05, 0.1]}
  ]

random_search_svm = RandomizedSearchCV(estimator=clf_svm, param_distributions=param_svm, cv=3, scoring='accuracy', refit = True, verbose = 3)
random_search_svm.fit(X_train_prepared, y_train)
#Best parameters
print(random_search_svm.best_params_)
print(random_search_svm.best_estimator_)

random_search_svm.cv_results_

best_model= random_search_svm.best_estimator_

X_test_prepared = full_pipeline.transform(X_test)
#predict using the best model
y_test_pred = best_model.predict(X_test_prepared)

from sklearn.metrics import accuracy_score
print("After Tuning:")
print("Accuracy", accuracy_score(y_test, y_test_pred))
print("Precision", precision_score(y_test, y_test_pred))
print("Recall", recall_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
