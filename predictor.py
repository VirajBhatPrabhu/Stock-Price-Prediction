#!/usr/bin/env python
# coding: utf-8

# #### Import the Libraries

# In[28]:


import pandas as pd
import pandas_datareader as pdr
import numpy as np


# #### We will use Pandas datareader and get the data from Tiingo
# #### Convert this data into a CSV file

# In[3]:


df=pdr.get_data_tiingo('TCS',api_key='f40f48935a6e8a75b47e4d15b88d587c4c74fd3e')
df.to_csv('TCS.csv')


# #### Read the csv file. we will use the Close value  for this project

# In[10]:


df_tcs = pd.read_csv('TCS.csv',usecols=[1,2,3,4,5])
df_tcs.head()
df2=df_tcs.reset_index()['close']


# In[27]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(12,6))
plt.plot(df2)
plt.xlabel('Time in Days')
plt.ylabel('Close Value of TCS Stocks')
plt.show()


# #### We will be using LSTM  fo prediction and hence its important that we sclae our data

# In[22]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df2=scaler.fit_transform(np.array(df2).reshape(-1,1))


# #### Split the data into Train and Test

# In[27]:


training=int(len(df2)* 0.65)
test=len(df2)-training
training_data,test_data=df2[0:training,:],df2[training:len(df2),:1]


# #### Convert an array of values into a dataset matrix

# In[31]:


def create_dataset(dataset,timestep):
    X,Y=[],[]
    for i in range(len(dataset)-timestep-1):
        a=dataset[i:(i+timestep),0]
        X.append(a)
        Y.append(dataset[i + timestep,0])
    return np.array(X),np.array(Y)    


# #### Create a time-series dataset

# In[34]:


timestep=100
X_train, y_train= create_dataset(training_data,timestep) 
X_test, y_test= create_dataset(test_data,timestep)


# #### Reshape the data 

# In[39]:


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[40]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Denseential
from keras.layers import Dense,LSTM


# #### Lets create a stacked LSTM model

# In[41]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')


# In[42]:


model.summary()


# In[47]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)


# #### Lets do prediction for train and test sets

# In[48]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# #### We need to de-nomalize our datasets

# In[49]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# #### Lets find the RMSE values for both train and test predictions

# In[50]:


from sklearn.metrics import mean_squared_error
import math

print(math.sqrt(mean_squared_error(y_train,train_predict)))
print(math.sqrt(mean_squared_error(y_test,test_predict)))


# In[32]:


lookback=100
trainpredictplot=np.empty_like(df2)
trainpredictplot[:,:]=np.nan
trainpredictplot[lookback:(len(train_predict)+lookback),:]=train_predict


testpredictplot=np.empty_like(df2)
testpredictplot[:,:]=np.nan
testpredictplot[len(train_predict)+(lookback*2)+1:len(df2)-1,:]=test_predict

plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df2),label='Original data')
plt.plot(trainpredictplot,label='Training data')
plt.plot(testpredictplot,label='Predicted stocks/Test data')
plt.legend(loc = 'upper right')
plt.xlabel('Time in Days')
plt.ylabel('Close Value of TCS Stocks')
plt.show()

                  
                  
                  
                  


# In[53]:


len(test_data)


# In[64]:


x_input=test_data[340:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# #### We wll do the prediction for next 30 days

# In[70]:


lst_output=[]
n_steps=100
i=0
while(i<30):
    if len(temp_input)>100:
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
        
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
        


# In[71]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[72]:


len(df2)


# In[33]:


plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df2[1157:]),label='last 100 days')
plt.plot(day_pred,scaler.inverse_transform(lst_output),label='Prediction next 30 days')
plt.legend(loc = 'upper right')
plt.xlabel('Time in Days')
plt.ylabel('Close Value of TCS Stocks')
plt.show()


# In[34]:


df3=df2.tolist()
plt.figure(figsize=(12,6))
df3.extend(lst_output)
plt.plot(df3[1200:])
plt.xlabel('Time in Days')
plt.ylabel('Close Value of TCS Stocks')
plt.show()


# In[35]:


df3=scaler.inverse_transform(df3).tolist()
plt.figure(figsize=(12,6))
plt.plot(df3)
plt.xlabel('Time in Days')
plt.ylabel('Close Value of TCS Stocks')
plt.show()


# In[38]:


last_val = test_predict[-1]
last_val_scaled = last_val/last_val
next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
print("Last Day Value:", np.asscalar(last_val))
print("Next Day Value:", np.asscalar(last_val*next_val)) 
print(np.append(last_val, next_val))

