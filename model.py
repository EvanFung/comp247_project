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


data = pd.read_csv('KSI.csv')
#printing percentage of missing values for each feature
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

#We noted that if the value of these columns are nan which means no
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

clean_data['PEDESTRIAN'] = np.where(clean_data['TRSN_CITY_VEH'] == 'Yes',1,0)
clean_data['PASSENGER'] = np.where(clean_data['TRSN_CITY_VEH'] == 'Yes',1,0)

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
clean_data['DISTRICT'] = clean_data['DISTRICT'].fillna('Other')
clean_data['VISIBILITY'] = clean_data['VISIBILITY'].fillna('Other')
clean_data['RDSFCOND'] = clean_data['RDSFCOND'].fillna('Other')

objdtype_cols = clean_data.select_dtypes(["object"]).columns
clean_data[objdtype_cols] = clean_data[objdtype_cols].astype('category')

clean_data['LATITUDE']=clean_data['LATITUDE'].astype('int')
clean_data['LONGITUDE']=clean_data['LATITUDE'].astype('int')


clean_data_target = data['ACCLASS']
clean_data_target = np.where(clean_data_target == 'Fatal',1,0)

clean_data = pd.get_dummies(clean_data, columns=['VISIBILITY','RDSFCOND','DISTRICT','LIGHT'])

