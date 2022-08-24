#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas_datareader as pdr
import numpy as np
import matplotlib.pyplot as plt 


# In[2]:


start='2010-01-01'
end='2022-08-23'
df=pdr.DataReader('AAPL','yahoo',start=start,end=end)
df.head()


# In[3]:


df.shape


# In[4]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.xlabel('Time in Days')
plt.ylabel('Close Value of Stocks')
plt.show()


# In[5]:


ma100=df['Close'].rolling(100).mean()
ma200=df['Close'].rolling(200).mean()


# In[6]:


import matplotlib.pyplot as plt 
plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.plot(ma100,'r',label='100 Days Moving Average')
plt.plot(ma200,'g',label='200 Days Moving Average')
plt.legend(loc='upper left')

plt.xlabel('Time')
plt.ylabel('Close Value of TCS Stocks')
plt.show()


# In[7]:


training_data = pd.DataFrame(df.Close[0:int(len(df)*0.70)])
test_data = pd.DataFrame(df.Close[int(len(df)*0.70): int(len(df))])


# In[8]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

training_data_array=scaler.fit_transform(training_data)


# In[9]:


X_train = []
y_train = []
for i in range(100, training_data_array.shape[0]):
    X_train.append(training_data_array[i-100: i])
    y_train.append(training_data_array[i, 0])
    
X_train,y_train=np.array(X_train),np.array(y_train)


# In[10]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


# In[11]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1),activation='relu'))
model.add(Dropout(0.2))

model.add(LSTM(60,return_sequences=True,activation='relu'))
model.add(Dropout(0.3))

model.add(LSTM(80,return_sequences=True,activation='relu'))
model.add(Dropout(0.4))

model.add(LSTM(120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')


# In[12]:


model.summary()


# In[13]:


model.fit(X_train,y_train,epochs=50)


# In[14]:


model.save('Tf_model.h5')


# In[15]:


last100=training_data.tail(100)
final_df=last100.append(test_data,ignore_index=True)


# In[17]:


input_data=scaler.fit_transform(final_df)


# In[18]:


X_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    X_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
X_test,y_test=np.array(X_test),np.array(y_test)


# In[19]:


y_predicted=model.predict(X_test)


# In[20]:


scaler.scale_


# In[21]:


scalefactor= 1/0.00682769
y_predicted= y_predicted*scalefactor
y_test=y_test*scalefactor


# In[23]:


plt.figure(figsize=(12,6))
plt.plot(y_predicted,'b',label='Oiginal Value')
plt.plot(y_test,'g',label='Predicted Value')
plt.xlabel('Time')
plt.ylabel('Close Value')

plt.legend(loc='upper left')
plt.show()


# In[ ]:




