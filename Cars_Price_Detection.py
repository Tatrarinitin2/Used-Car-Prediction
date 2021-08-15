#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore') 


# In[2]:


#Loading data set
data=pd.read_csv('cars.csv')
data


# In[3]:


# Uploading data in Data frame
df=pd.DataFrame(data)
df.head()


# In[4]:


#Checking data type
df.dtypes


# In[5]:


#Checking duplicated rows
df.duplicated().sum()


# In[6]:


#Checking unique values, value counts & null values
for i in df.columns:
    print('\n')
    print(i)
    print(df[i].unique())
    print('The Total no. of unique value:',len(df[i].unique()))
    print(df[i].value_counts())
    print(df[i].isnull().sum())


# In[7]:


# Dropping 'Unnamed: 0' as it show index value
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()


# In[8]:


# Keeping correct data
df=df[df['FUEL'].isin(['Petrol','Diesel','CNG & Hybrids','LPG','Electric'])]


# In[9]:


#  Keeping correct data
df = df[df['TRANSMISSION'].isin(['Manual','Automatic'])]


# In[10]:


#Removing '-' from the data frame 
for i in df.columns:
    df = df[df[i] != '-']


# In[11]:


for i in df.columns:
    print('\n')
    print(i)
    print(df[i].unique())
    print('The Total no. of unique value:',len(df[i].unique()))
    print(df[i].value_counts())
    print(df[i].isnull().sum())


# In[12]:


# Renaming data
df['YEAR']=df['YEAR'].replace({'2011.0':'2011','2015.0':'2015','2012.0':'2012','2014.0':'2014'})


# Keeping all Brand having value count more than 20 

# In[13]:


s=df.BRAND.value_counts().gt(20)
df=df.loc[df.BRAND.isin(s[s].index)]


# In[14]:


s=df.MODEL.value_counts().gt(10)
df=df.loc[df.MODEL.isin(s[s].index)]


# In[15]:


for i in df.columns:
    print('\n')
    print(i)
    print(df[i].unique())
    print('The Total no. of unique value:',len(df[i].unique()))
    print(df[i].value_counts())
    print(df[i].isnull().sum())


# Keeping all MODEL having value count more than 10 

# In[16]:


s=df.MODEL.value_counts().gt(10)
df=df.loc[df.MODEL.isin(s[s].index)]


# In[17]:


for i in df.columns:
    print('\n')
    print(i)
    print(df[i].unique())
    print('The Total no. of unique value:',len(df[i].unique()))
    print(df[i].value_counts())
    print(df[i].isnull().sum())


# Since VARIANT have 1306 unique values, so we drop it.

# In[18]:


df.drop('VARIANT', axis=1, inplace=True)


# In[19]:


df.head()


# Splitting LOCATION into state name only

# In[20]:


df2=df['LOCATION'].str.split(',',n=2,expand=True)


# In[21]:


df2.head()


# In[22]:


df2.drop({0,1},axis=1,inplace=True)


# In[96]:


df2.head()


# In[24]:


df2.rename(columns={2:'STATE'},inplace=True)


# In[25]:


df=pd.concat([df,df2],axis=1)


# In[26]:


df.drop('LOCATION',axis=1, inplace=True)


# In[27]:


df['STATE'].unique()


# In[28]:


df['STATE'].value_counts()


# In[29]:


#Correct data
df['STATE']=df['STATE'].replace({' Pune, Maharashtra':' Maharashtra',' Rajgarh, Rajasthan':' Rajasthan',
                                 ' India':' Rajasthan'})


# In[30]:


df.head()


# Removing '₹' and ',' from PRICE

# In[31]:


df['PRICE'] = df['PRICE'].str.replace('₹','').str.replace(',','')


# Removing 'KM' and ',' from 'KM_Travelled' column

# In[32]:


df['KM_Travelled'] = df['KM_Travelled'].str.replace(',','').str.replace('km','')


# In[33]:


df.head()


# In[34]:


#Changing data type to float
df[['YEAR','KM_Travelled','PRICE']]=df[['YEAR','KM_Travelled','PRICE']].astype(float)


# In[35]:


df.dtypes


# In[36]:


df.describe()


# There are outliers in data frame

# In[37]:


df.corr()


# KM_travelled has low correlation with target variable

# In[38]:


#Checking shape of data 
df.shape


# Treating Outliers with IQR method

# In[39]:


for i in ['YEAR','KM_Travelled','PRICE']:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    df = df[((df[i] > (Q1 - 1.5 * IQR)) & (df[i] < (Q3 + 1.5 * IQR)))]


# In[40]:


df.shape


# In[41]:


for i in df.columns:
    print('\n')
    print(i)
    print(df[i].unique())
    print('The Total no. of unique value:',len(df[i].unique()))
    print(df[i].value_counts())
    print(df[i].isnull().sum())


# SInce there is only one entry with 'Electriv' as fuel, we remove it

# In[42]:


df.drop(df.index[df['FUEL']=='Electric'],inplace=True)


# In[43]:


df.shape


# In[44]:


s=df.BRAND.value_counts().gt(20)
df=df.loc[df.BRAND.isin(s[s].index)]


# In[45]:


s=df.MODEL.value_counts().gt(10)
df=df.loc[df.MODEL.isin(s[s].index)]


# In[46]:


for i in df.columns:
    print('\n')
    print(i)
    print(df[i].unique())
    print('The Total no. of unique value:',len(df[i].unique()))
    print(df[i].value_counts())
    print(df[i].isnull().sum())


# In[47]:


s=df.BRAND.value_counts().gt(20)
df=df.loc[df.BRAND.isin(s[s].index)]


# In[48]:


df.shape


# In[49]:


df.corr()


# In[50]:


df.describe()


# Outlier have reduced

# In[51]:


sns.pairplot(df)


# In[52]:


plt.figure(figsize=(15,10))
sns.countplot(x='PRICE',hue='FUEL',data=df)
plt.show()


# CArs which run on Diesels are more expensive

# In[53]:


plt.figure(figsize=(15,5))
sns.countplot(df['BRAND'])
plt.show()


# Maruti Suzuki Brand car are more available for re-sell 

# In[54]:


plt.figure(figsize=(12,9))
sns.scatterplot(x='KM_Travelled',y='PRICE',hue='FUEL',data=df)
plt.show()


# In[55]:


plt.figure(figsize=(12,9))
sns.scatterplot(x='KM_Travelled',y='PRICE',hue='TRANSMISSION',data=df)
plt.show()


# Manula Transmission vehicle have travelled more distance than Automatic

# In[56]:


plt.figure(figsize=(12,9))
sns.scatterplot(x='KM_Travelled',y='PRICE',hue='No._of_owners',data=df)
plt.show()


# 1S townership vehicle are expensive than 2nd ownership

# In[58]:


plt.figure(figsize=(12,9))
sns.scatterplot(x='KM_Travelled',y='PRICE',hue='STATE',data=df)
plt.show()


# There is no relation of state with price, thus we can drop it

# In[59]:


df.drop('STATE',axis=1,inplace=True)


# In[60]:


plt.figure(figsize=(12,9))
sns.scatterplot(x='YEAR',y='PRICE',hue='TRANSMISSION',data=df)
plt.show()


# Vehicle with automatic Transmission are comparitively more expensive than Manual

# In[61]:


plt.figure(figsize=(12,9))
sns.scatterplot(x='YEAR',y='PRICE',hue='No._of_owners',data=df)
plt.show()


# 1st hand cars are more expensive than 2nd hand

# We will convert the Brand name with the value of it count

# In[62]:


df_frequency_brand=df.BRAND.value_counts().to_dict()


# In[63]:


df.BRAND=df.BRAND.map(df_frequency_brand)


# In[64]:


df.head()


# We will convert the Model name with the value of it count

# In[65]:


df_frequency_model=df.MODEL.value_counts().to_dict()


# In[66]:


df.MODEL=df.MODEL.map(df_frequency_model)


# In[67]:


df.head()


# ### Categorical coding with Label Encoding

# In[68]:


from sklearn import preprocessing


# In[69]:


le=preprocessing.LabelEncoder()


# In[70]:


for i in ['FUEL','TRANSMISSION','No._of_owners']:
    df[i]=le.fit_transform(df[i])


# In[71]:


df.head()


# In[72]:


#Splitting the data


# In[73]:


y=df['PRICE']


# In[74]:


x=df.drop('PRICE',axis=1)


# In[75]:


y.shape


# In[76]:


x.shape


# ### Standarizing the data

# In[77]:


from sklearn.preprocessing import StandardScaler


# In[78]:


sc=StandardScaler()
x=sc.fit_transform(x)


# ## MODEL SELECTION

# In[79]:


from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso,Ridge


# In[80]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[81]:


lg=LinearRegression()
lg.fit(x_train,y_train)
pred=lg.predict(x_test)
print("R2_score:",r2_score(y_test,pred))
print('mean_squared_error:',mean_squared_error(y_test,pred))
print('mean_absolute_error:',mean_absolute_error(y_test,pred))
print('Root_mean_square_error:',np.sqrt(mean_squared_error(y_test,pred)))


# In[82]:


lg.score(x_train,y_train)


# In[83]:


lg.score(x_test,y_test)


# In[84]:


dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
pred=dtr.predict(x_test)
print("R2_score:",r2_score(y_test,pred))
print('mean_squared_error:',mean_squared_error(y_test,pred))
print('mean_absolute_error:',mean_absolute_error(y_test,pred))
print('Root_mean_square_error:',np.sqrt(mean_squared_error(y_test,pred)))


# In[85]:


ls=Lasso(alpha=0.0001)
ls.fit(x_train,y_train)


# In[86]:


ls.score(x_train,y_train)


# In[97]:


lg.score(x_test,y_test)


# In[87]:


svr=SVR()
svr.fit(x_train,y_train)
pred=svr.predict(x_test)
print("R2_score:",r2_score(y_test,pred))
print('mean_squared_error:',mean_squared_error(y_test,pred))
print('mean_absolute_error:',mean_absolute_error(y_test,pred))
print('Root_mean_square_error:',np.sqrt(mean_squared_error(y_test,pred)))


# In[88]:


kn=KNeighborsRegressor()
kn.fit(x_train,y_train)
pred=kn.predict(x_test)
print("R2_score:",r2_score(y_test,pred))
print('mean_squared_error:',mean_squared_error(y_test,pred))
print('mean_absolute_error:',mean_absolute_error(y_test,pred))
print('Root_mean_square_error:',np.sqrt(mean_squared_error(y_test,pred)))


# ## ENSEMBLE TECHNIQUE

# In[89]:


from sklearn.ensemble import RandomForestRegressor


# In[90]:


rfr=KNeighborsRegressor()
rfr.fit(x_train,y_train)
pred=rfr.predict(x_test)
print("R2_score:",r2_score(y_test,pred))
print('mean_squared_error:',mean_squared_error(y_test,pred))
print('mean_absolute_error:',mean_absolute_error(y_test,pred))
print('Root_mean_square_error:',np.sqrt(mean_squared_error(y_test,pred)))


# ## HYPERPARAMETER TUNNING

# In[91]:


from sklearn.model_selection import GridSearchCV


# In[92]:


crit={'criterion':['mse','friedman_mse','mae','poisson']}
grid=GridSearchCV(estimator=dtr,param_grid=crit)
grid.fit(x,y)
print(grid)
print(grid.best_score_)
print(grid.best_estimator_.criterion)
print(grid.best_params_)


# In[93]:


dtr=DecisionTreeRegressor(criterion='mse')
dtr.fit(x_train,y_train)
pred=dtr.predict(x_test)
print("R2_score:",r2_score(y_test,pred))
print('mean_squared_error:',mean_squared_error(y_test,pred))
print('mean_absolute_error:',mean_absolute_error(y_test,pred))
print('Root_mean_square_error:',np.sqrt(mean_squared_error(y_test,pred)))


# ## Saving the best model

# In[94]:


import joblib


# In[95]:


joblib.dump(dtr,'cars.pkl')

