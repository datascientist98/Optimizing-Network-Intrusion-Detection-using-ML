#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tkinter as tk
from tkinter import *
from tkinter.messagebox import *
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras.layers.core import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
import csv


# In[3]:


#Step 1: Data Collection
df_train = pd.read_csv(r'C:\Users\Sara\Desktop\final year project\UNSW_NB15_training-set.csv')
df_train.head()


# In[4]:


df_test = pd.read_csv(r'C:\Users\Sara\Desktop\final year project\UNSW_NB15_testing-set.csv')
df_test.head()


# In[5]:


df = pd.concat([df_train, df_test])
df.head()
print (len(df))


# In[6]:


#Handle missing values by replacing the missing value with -99999
df.replace('-',-99999, inplace=True)
df.head(20)


# In[7]:


#Handle missing values by replacing the missing value with -99999
df.replace('no',-99999, inplace=True)
df.head(20)


# In[8]:


#Convert the nominal (categorial) attributes to numerical attributes
#converting 'protocol' feature to numerical value
lab_enc = LabelEncoder()
df['proto'] = lab_enc.fit_transform(df['proto'].astype('str'))
df.head()


# In[9]:


#converting 'service' feature to numerical value
df['service'].replace('ftp', 1 ,inplace=True)
df['service'].replace('smtp', 2 ,inplace=True)
df['service'].replace('snmp', 3 ,inplace=True)
df['service'].replace('http', 4 ,inplace=True)
df['service'].replace('ftp-data', 5 ,inplace=True)
df['service'].replace('dns', 6 ,inplace=True)
df['service'].replace('ssh', 7 ,inplace=True)
df['service'].replace('radius', 8 ,inplace=True)
df['service'].replace('pop3', 9 ,inplace=True)
df['service'].replace('dhcp', 10 ,inplace=True)
df['service'].replace('ssl', 11 ,inplace=True)
df['service'].replace('irc', 12 ,inplace=True)
df.head()


# In[10]:


#converting 'state' feature to numerical value
lab_enc = LabelEncoder()
df['state'] = lab_enc.fit_transform(df['state'].astype('str'))
df.head()


# In[11]:


#converting 'attack_cat' feature to numerical value
df['attack_cat'].replace('Normal', 0 ,inplace=True)
df['attack_cat'].replace('Fuzzers', 1 ,inplace=True)
df['attack_cat'].replace('Analysis', 2 ,inplace=True)
df['attack_cat'].replace('Backdoor', 3 ,inplace=True)
df['attack_cat'].replace('DoS', 4 ,inplace=True)
df['attack_cat'].replace('Exploits', 5 ,inplace=True)
df['attack_cat'].replace('Generic', 6 ,inplace=True)
df['attack_cat'].replace('Reconnaissance', 7 ,inplace=True)
df['attack_cat'].replace('Shellcode', 8 ,inplace=True)
df['attack_cat'].replace('Worms', 9 ,inplace=True)
df.head()


# In[12]:


#random shuffling of dataset 
df = df.sample(frac=1).reset_index(drop=True)
df.head()


# In[13]:


columns = ['id','attack_cat','label']
df_x = df.drop(columns, axis = 1)
df_x.head()


# In[14]:


df_y = df.iloc[:,-1]
df_y


# In[15]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[16]:


df_x_norm = normalize(df_x)
df_x_norm


# In[17]:


print(len(df_x_norm))
print(len(df_y))


# In[18]:


df_x_train = df_x_norm.iloc[0:154603 , :]
df_x_train.head()


# In[19]:


print(len(df_x_train))


# In[20]:


df_x_val = df_x_norm.iloc[154603: 206136, :]
df_x_val.head()


# In[21]:


print(len(df_x_val))


# In[22]:


df_x_test = df_x_norm.iloc[206136:257673, :]
df_x_test.head()


# In[23]:


print(len(df_x_test))


# In[24]:


df_y_train = df_y.iloc[0:154603]
df_y_train.head()


# In[25]:


print(len(df_y_train))


# In[26]:


df_y_val = df_y.iloc[154603:206136]
df_y_val.head()


# In[27]:


print(len(df_y_val))


# In[28]:


df_y_test = df_y.iloc[206136:257673]
df_y_test.head()


# In[29]:


print(len(df_y_test))


# In[30]:


X_train = df_x_train.to_numpy()


# In[31]:


X_train


# In[32]:


X_test = df_x_test.to_numpy()


# In[33]:


X_test


# In[34]:


Y_train = df_y_train.to_numpy()


# In[35]:


Y_train


# In[36]:


Y_test = df_y_test.to_numpy()


# In[37]:


Y_test


# In[38]:


X_val = df_x_val.to_numpy()
X_val


# In[39]:


Y_val = df_y_val.to_numpy()
Y_val


# In[45]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, Y_train)


# In[46]:


y_pred = svclassifier.predict(X_test)


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))


# In[48]:


from sklearn.metrics import accuracy_score
acc_score = accuracy_score(Y_test, y_pred)
acc_percentage = acc_score * 100
acc_percentage


# In[ ]:





# In[ ]:




