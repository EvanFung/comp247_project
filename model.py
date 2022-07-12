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
drop_column = ['OFFSET','FATAL_NO','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND','CYCLIST','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','REDLIGHT','ALCOHOL','DISABILITY','ObjectId','X','Y','INDEX_']
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
data.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True) # Neighbourhood is identical with Hood ID
# change data type
data['DATE'] = pd.to_datetime(data['DATE'])
data['DAY'] = pd.to_datetime(data['DATE']).dt.day
data['MONTH'] = data['DATE'].dt.month
data['MINUTE'],data['SECOND'] = divmod(data['TIME'], 60)
df_timestamp = pd.DataFrame({'year':data['YEAR'],'month':data['MONTH'],'day':data['DAY'],'hour':data['HOUR'],'minute':data['MINUTE'],'second':data['SECOND']})
data['TIMESTAMP'] = pd.to_datetime(df_timestamp)

#may delete date, hours, minutes.. later
objdtype_cols = data.select_dtypes(["object"]).columns
data[objdtype_cols] = data[objdtype_cols].astype('category')
data.info()

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