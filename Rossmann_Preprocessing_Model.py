#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import warnings
warnings.filterwarnings("ignore")

#import the required csv files
df_train = pd.read_csv("train.csv")
df_store = pd.read_csv("store.csv")
df_test = pd.read_csv("test.csv")

#Merge the training and store csv 
df_train = pd.merge(df_train, df_store, how = "inner", on = "Store")
df_test = pd.merge(df_test, df_store, how = "inner", on = "Store")


# In[2]:


def PreProcessing(df, a_string):
    #Split the Date into a year a month and a date column in general
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format = True)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    #print(df_train.iloc[:,-3:].head())

    #Check if there are any Nan values in the data set 
    #print(df_train.isnull().sum())
    # For competitionDistance, use the mean of the column to fill any NaN values
    # For months, years and categorical data, use the most occuring value to fill NaN values
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].mean())
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(df["CompetitionOpenSinceMonth"].value_counts().idxmax())
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(df["CompetitionOpenSinceYear"].value_counts().idxmax())
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(df["Promo2SinceWeek"].value_counts().idxmax())
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(df["Promo2SinceYear"].value_counts().idxmax())
    df["PromoInterval"] = df["PromoInterval"].fillna(df["PromoInterval"].value_counts().idxmax())

    #df_train.head()
    df.drop(columns=["Date"], axis = 1, inplace = True)

    #Fully categorising the StateHoliday Column
    df["StateHoliday"] = df["StateHoliday"].replace({0: "n"})

    #Label encode all the categorical columns. 
    CategoricalColumns = ["StoreType", "Assortment", "PromoInterval", "StateHoliday"]
    le = LabelEncoder()
    for Column in CategoricalColumns:
        df[Column] = le.fit_transform(df[Column])
   

    # Convert year and months columns into a single column of just months
    df["CompetitionOpen"] = 12 * (df["Year"] - df["CompetitionOpenSinceYear"]) + (df["Month"] - df["CompetitionOpenSinceMonth"])
    df["PromoOpen"] = 12 * (df["Year"] - df["Promo2SinceYear"]) + (df["WeekOfYear"] - df["Promo2SinceWeek"]) / 4

    # Remove Customers Column if contained in the dataframe
    for column, data in df.items():
        if column == "Customers": 
            df.drop(columns=["Customers"], axis = 1, inplace = True)
        else:
            continue

    if a_string == "df_train":
      df = df.loc[df["Open"] != 0]


    df.drop(columns=["Open"], axis = 1, inplace = True)
    # Drop newly useless columns
    df.drop(columns=["CompetitionOpenSinceYear"], axis = 1, inplace = True)
    df.drop(columns=["CompetitionOpenSinceMonth"], axis = 1, inplace = True)
    df.drop(columns=["Promo2SinceWeek"], axis = 1, inplace = True)
    df.drop(columns=["Promo2SinceYear"], axis = 1, inplace = True)

    #Normalize the data in every column
    #scaler=MinMaxScaler()
    #col = df.columns
    #result=scaler.fit_transform(df)
    #df = pd.DataFrame(result, columns = col)
    return df

df_train = PreProcessing(df_train, "train")
df_test = PreProcessing(df_test, "test")
#df_train.to_csv("output_file.csv", index=False, encoding="utf8")
#df_test.to_csv("testing_file.csv", index=False, encoding="utf8")


# In[3]:


df_train


# In[4]:


df_test


# In[5]:


ytrain=df_train['Sales']
xtrain=df_train.drop('Sales',axis=1)


# In[6]:


ytrain


# In[7]:


xtrain


# In[8]:


df_test=df_test.drop('Id',axis=1)


# In[9]:


scaler=MinMaxScaler(feature_range=(0,1))
col = xtrain.columns
result_train_x=scaler.fit_transform(xtrain)
result_test=scaler.fit_transform(df_test)


# In[10]:


df_y_train = pd.DataFrame(ytrain, columns =['Sales'] )


# In[11]:


scaler_y=MinMaxScaler(feature_range=(0,1))
result_train_y=scaler_y.fit_transform(df_y_train[['Sales']])


# In[12]:


X_train, X_val, y_train, y_val = train_test_split(result_train_x, result_train_y, test_size=0.20, random_state=7)


# In[13]:


df_y_train


# In[14]:


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding


model = Sequential([
  LSTM(units = 2, return_sequences = True, input_shape = (X_train.shape[1], 1)),
  Dropout(0.3),
  LSTM(units =2, return_sequences = True),
  Dropout(0.3),
  LSTM(units = 2),
  Dense(1),
])


# In[15]:


model.summary()


# In[16]:


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])


# In[17]:


hist = model.fit(X_train, y_train,
          batch_size=256, epochs=2,
          validation_data=(X_val, y_val))


# In[18]:


testPredict = model.predict(result_test)


# In[19]:


testPredict


# In[20]:


inv_yhat=scaler_y.inverse_transform(testPredict)


# In[21]:


inv_yhat


# In[22]:


testPredict = pd.DataFrame(data=inv_yhat, columns=["Sales"])
testPredict.insert(0, 'ID', range(1, 1 + len(testPredict)))
testPredict


# In[23]:


testPredict.to_csv('predictions.csv', index=False)

