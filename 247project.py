# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 06:10:27 2022

@author: dijia
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

data = pd.read_csv('KSI.csv')
print("Initial shape:", data.shape)

# There are several columns consist of "Yes" and "<Null>" (where Null means No). 
# For these binary column, replace  "<Null>" with"No"
binary_cols = ['CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','REDLIGHT','ALCOHOL','DISABILITY','PASSENGER','AG_DRIV','PEDESTRIAN']
data[binary_cols]=data[binary_cols].replace({'<Null>':0, 'Yes':1})

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
# Police DIVISION is identical with DIVISION
# X,Y are longitude and latitudes, dulicate, drop X and Y
data.drop(['NEIGHBOURHOOD','POLICE_DIVISION','X','Y'], axis=1, inplace=True)

# remove other irrelevant columns or columns contain to many missing values
data.drop(['MANOEUVER','DRIVACT','DRIVCOND','INITDIR','STREET1','STREET2','WARDNUM'], axis=1, inplace=True)

#"""#Visualization"""----------------------------------------------------------------------------------------------------

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

# Where does accident more likly to occur
#2D histogram
plt.hist2d(data['LATITUDE'], data['LONGITUDE'], bins=(40, 40), cmap=plt.cm.jet)
plt.title("2D histogram of all fatal and non-fatal accidents")
plt.show()

data_Fatal = data[data['ACCLASS'] == 'Fatal']
plt.hist2d(data_Fatal['LATITUDE'], data_Fatal['LONGITUDE'], bins=(40, 40), cmap=plt.cm.jet)
plt.title("2D histogram of fatal accidents")
plt.show()


# scatter plot of all fatal and non-fatal accidents
sns.scatterplot(x='LATITUDE', y='LONGITUDE', data = data, hue = "ACCLASS",alpha=0.3)
plt.title("Accidents")
plt.show()
#scatter plot of fatal accidents
sns.scatterplot(x='LATITUDE', y='LONGITUDE', data = data[data['ACCLASS'] == 'Fatal'],alpha=0.3)
plt.title("Fatal Accidents")
plt.show()

data.shape
data.dtypes

#"""#Further Data Cleaning--------------------------------------------------------------------------------------------"""

#ACCNUM is identifier, drop
data.drop(['ACCNUM'], axis=1, inplace=True)

#drop rows that contain missing value
data.dropna(subset=['ROAD_CLASS', 'DISTRICT','VISIBILITY','RDSFCOND','LOCCOORD','IMPACTYPE','DIVISION','TRAFFCTL','INVTYPE'],inplace=True)

#target class
data['ACCLASS']=data['ACCLASS'].replace({'Non-Fatal':0, 'Fatal':1})
data['ACCLASS'].value_counts()   #dataset is unbalanced

#Test Train split
#Since the dataset is unbalanced, use straified split
X = data.drop(["ACCLASS"], axis=1)
y= data["ACCLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5,stratify=y)

#impute
imputer = SimpleImputer(strategy="constant",fill_value='missing')  
data_tr=imputer.fit_transform(X_train)
data_tr= pd.DataFrame(data_tr, columns=X_train.columns)

print(data_tr.isna().sum()/len(data_tr)*100)

#numerical features
num_columns=['YEAR', 'TIME', 'HOUR', 'LATITUDE', 'LONGITUDE', 
       'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
       'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV',
       'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'HOOD_ID', 'WEEKDAY', 'DAY',
       'MONTH']
data_num =data_tr[num_columns] 
#data_num = data_tr.select_dtypes(include=[np.number])
print(data_num.columns)

#categorical features
cat_columns=['ROAD_CLASS', 'DISTRICT', 'DIVISION', 'LOCCOORD', 'ACCLOC', 'TRAFFCTL',
       'VISIBILITY', 'LIGHT', 'RDSFCOND','IMPACTYPE', 'INVTYPE',
       'INVAGE', 'INJURY', 'VEHTYPE']
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

X_train_prepared.shape

"""#Feature Selection"""

#method 1: using ExtraTreesClassifier for feature selection
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train_prepared,y_train)
#print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=df.columns)

#plot the most important 30 features 
fig = plt.gcf()
fig.set_size_inches(12, 8)
feat_importances.nlargest(30).plot(kind='barh')
plt.show()

#method 2: using SelectFromModel and RandomForestClassifier to select features
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
sel.fit(X_train_prepared, y_train)
selected_feat= df.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)

feat_importances = pd.Series(sel.estimator_.feature_importances_, index=df.columns)
#plot the selected features
fig = plt.gcf()
fig.set_size_inches(12, 8)
feat_importances.nlargest(40).plot(kind='barh')
plt.title("Feature Importance")
plt.show()

#"""# Model Training and Testing""" --------------------------------------------------------------------------------

df_X_train_prepared= pd.DataFrame(X_train_prepared.toarray(), columns=df.columns)

#Logistic Regression 
from sklearn.linear_model import LogisticRegression
LR_clf= LogisticRegression()
selected=['YEAR', 'TIME', 'HOUR', 'LATITUDE', 'LONGITUDE', 'PEDESTRIAN',
       'AUTOMOBILE', 'TRUCK', 'TRSN_CITY_VEH', 'PASSENGER', 'SPEEDING',
       'AG_DRIV', 'REDLIGHT', 'HOOD_ID', 'WEEKDAY', 'DAY', 'MONTH',
       'ROAD_CLASS_Major Arterial', 'ROAD_CLASS_Minor Arterial',
       'DISTRICT_North York', 'DISTRICT_Toronto and East York',
       'LOCCOORD_Intersection', 'LOCCOORD_Mid-Block',
       'ACCLOC_Non Intersection', 'ACCLOC_missing', 'TRAFFCTL_Traffic Signal',
       'VISIBILITY_Rain', 'LIGHT_Dark, artificial', 'LIGHT_Daylight',
       'RDSFCOND_Wet', 'IMPACTYPE_Approaching',
       'IMPACTYPE_Pedestrian Collisions', 'IMPACTYPE_Rear End',
       'IMPACTYPE_SMV Other', 'IMPACTYPE_Turning Movement', 'INVTYPE_Driver',
       'INVTYPE_Passenger', 'INVTYPE_Pedestrian', 'INVAGE_25 to 29',
       'INVAGE_unknown', 'INJURY_Major', 'INJURY_Minimal', 'INJURY_Minor',
       'INJURY_None', 'VEHTYPE_Other', 'VEHTYPE_missing']
X_train_selected=df_X_train_prepared[selected]
LR_clf.fit(X_train_selected, y_train)
#accuracy on training dataset
LR_clf.score(X_train_selected,y_train)

#test
X_test_prepared = full_pipeline.transform(X_test)
df_X_test_prepared= pd.DataFrame(X_test_prepared.toarray(), columns=df.columns)  #transform the test dataset into dataframe, then slice the obtained dataset with selected features
y_test_pred=LR_clf.predict(df_X_test_prepared[selected])

from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test, y_test_pred)
print(classification_report(y_test, y_test_pred))
