#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#Step 1: Data Collection
df_train = pd.read_csv(r'C:\Users\Sara\Desktop\final year project\UNSW_NB15_training-set.csv')
df_train.head()


# In[3]:


root = tk.Tk()
root.geometry('1100x600')
root.title("Intrusion Detection System Using Machine Learning")
label_0 = Label(root, text="Intrusion Detection System",width=20,font=("bold", 20))
label_0.grid(row = 1 , column = 2)
label_1 = Label(root, text="Upload the csv file here",width=20,font=("bold", 10))
label_1.grid(row = 5 , column = 0)

def import_csv_data():
    global v
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)
    df = pd.read_csv(csv_file_path)
    print(df.head())


# In[4]:


tk.Label(root, text='File Path').grid(row=9, column=1)
v = tk.StringVar()
entry = tk.Entry(root, textvariable=v).grid(row=10, column=1)
tk.Button(root, text='Browse Data Set',command=import_csv_data).grid(row=11, column=1)
tk.Button(root, text='Upload',command=root.destroy).grid(row=11, column=2)
root.mainloop()


# In[5]:


df_test = pd.read_csv(r'C:\Users\Sara\Desktop\final year project\UNSW_NB15_testing-set.csv')
df_test.head()


# In[6]:


df = pd.concat([df_train, df_test])
df.head()
print (len(df))


# In[7]:


#Step 2: Data Preprocessing
#Getting unique values of nominal attributes
df.proto.unique()


# In[8]:


df.service.unique()


# In[9]:


df.state.unique()


# In[10]:


df.attack_cat.unique()


# In[11]:


#checking datatype of features
for col in df:
    print (type(df[col][1]))


# In[12]:


df.head(20)


# In[13]:


#Handle missing values by replacing the missing value with -99999
df.replace('-',-99999, inplace=True)
df.head(20)


# In[14]:


#Handle missing values by replacing the missing value with -99999
df.replace('no',-99999, inplace=True)
df.head(20)


# In[15]:


#Convert the nominal (categorial) attributes to numerical attributes
#converting 'protocol' feature to numerical value
lab_enc = LabelEncoder()
df['proto'] = lab_enc.fit_transform(df['proto'].astype('str'))
df.head()


# In[16]:


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


# In[17]:


#converting 'state' feature to numerical value
lab_enc = LabelEncoder()
df['state'] = lab_enc.fit_transform(df['state'].astype('str'))
df.head()


# In[18]:


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


# In[19]:


#random shuffling of dataset 
df = df.sample(frac=1).reset_index(drop=True)
df.head()


# In[20]:


columns = ['id','attack_cat','label']
df_x = df.drop(columns, axis = 1)
df_x.head()


# In[21]:


df_y = df.iloc[:,-1]
df_y


# In[22]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[23]:


df_x_norm = normalize(df_x)
df_x_norm


# In[24]:


print(len(df_x_norm))
print(len(df_y))


# In[25]:


df_x_train = df_x_norm.iloc[0:154603 , :]
df_x_train.head()


# In[26]:


print(len(df_x_train))


# In[27]:


df_x_val = df_x_norm.iloc[154603: 206136, :]
df_x_val.head()


# In[28]:


print(len(df_x_val))


# In[29]:


df_x_test = df_x_norm.iloc[206136:257673, :]
df_x_test.head()


# In[30]:


print(len(df_x_test))


# In[31]:


df_y_train = df_y.iloc[0:154603]
df_y_train.head()


# In[32]:


print(len(df_y_train))


# In[33]:


df_y_val = df_y.iloc[154603:206136]
df_y_val.head()


# In[34]:


print(len(df_y_val))


# In[35]:


df_y_test = df_y.iloc[206136:257673]
df_y_test.head()


# In[36]:


print(len(df_y_test))


# In[37]:


X_train = df_x_train.to_numpy()


# In[38]:


X_train


# In[39]:


X_test = df_x_test.to_numpy()


# In[40]:


X_test


# In[41]:


Y_train = df_y_train.to_numpy()


# In[42]:


Y_train


# In[43]:


Y_test = df_y_test.to_numpy()


# In[44]:


Y_test


# In[45]:


X_val = df_x_val.to_numpy()
X_val


# In[46]:


Y_val = df_y_val.to_numpy()
Y_val


# In[47]:


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=42, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(12, activation='relu',  kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(1, activation='sigmoid'))


# In[48]:


# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[49]:


history = model.fit(X_train, Y_train, epochs=75, batch_size=10, validation_data=(X_val, Y_val), callbacks = [EarlyStopping(monitor='acc', patience=2)])


# In[50]:


# evaluate the keras model
_, accuracy = model.evaluate(X_train, Y_train)
print('Training Accuracy: %.2f' % (accuracy*100))


# In[51]:


# evaluate the keras model
_, accuracy1 = model.evaluate(X_test, Y_test)
print('Testing Accuracy: %.2f' % (accuracy1*100))


# In[52]:


prediction = model.predict_classes(X_test)
print(prediction)


# In[53]:


Y_test


# In[54]:


#understanding the sigmoid curve
input = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1/(1+np.exp(-x))

from matplotlib import pyplot as plt
plt.plot(input, sigmoid(input), c="r")
plt.show()


# In[55]:


X_test[0]


# In[56]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test,prediction))
print(classification_report(Y_test,prediction))


# In[ ]:





# In[ ]:




