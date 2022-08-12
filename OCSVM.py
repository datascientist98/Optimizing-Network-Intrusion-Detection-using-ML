#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from matplotlib import pyplot

import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
import csv
import sys
import os
from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from tkinter.messagebox import *


# In[2]:


#Step 1: Data Collection
df_train = pd.read_csv(r'C:\Users\Sara\Desktop\final year project\UNSW_NB15_training-set.csv')
df_train.head()


# In[3]:


df_test = pd.read_csv(r'C:\Users\Sara\Desktop\final year project\UNSW_NB15_testing-set.csv')
df_test.head()


# In[4]:


df = pd.concat([df_train, df_test])
df.head()
print (len(df))


# In[5]:


#Handle missing values by replacing the missing value with -99999
df.replace('-',-99999, inplace=True)
df.head(20)


# In[6]:


#Handle missing values by replacing the missing value with -99999
df.replace('no',-99999, inplace=True)
df.head(20)


# In[7]:


#Convert the nominal (categorial) attributes to numerical attributes
#converting 'protocol' feature to numerical value
lab_enc = LabelEncoder()
df['proto'] = lab_enc.fit_transform(df['proto'].astype('str'))
df.head()


# In[8]:


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


# In[9]:


#converting 'state' feature to numerical value
lab_enc = LabelEncoder()
df['state'] = lab_enc.fit_transform(df['state'].astype('str'))
df.head()


# In[10]:


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


# In[11]:


#random shuffling of dataset 
df = df.sample(frac=1).reset_index(drop=True)
df.head()


# In[12]:


df["label"].replace({0: 1, 1: -1}, inplace=True)
df


# In[13]:


normal = ['0']
nor_obs = df[df.attack_cat.isin(normal)]
nor_obs


# In[14]:


anomaly = ['1','2','3','4','5','6','7','8']
ano_obs = df[df.attack_cat.isin(anomaly)]
ano_obs


# In[15]:


print(len(nor_obs))


# In[16]:


nor_features = nor_obs.iloc[0:80000, :]
columns = ['id','attack_cat','label']
nor_features = nor_features.drop(columns, axis = 1)
nor_features


# In[17]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[18]:


x_train_1 = normalize(nor_features)
x_train_1


# In[19]:


nor_features_2 = nor_obs.iloc[80000:93000, :]
columns = ['id','attack_cat','label']
nor_features_2 = nor_features_2.drop(columns, axis = 1)
nor_features_2


# In[20]:


x_train_2 = normalize(nor_features_2)
x_train_2


# In[21]:


y_train_1 = nor_obs.iloc[0:80000,-1]
y_train_1


# In[22]:


print(len(y_train_1))


# In[23]:


y_train_2 = nor_obs.iloc[80000:93000,-1]
y_train_2


# In[24]:


print(len(y_train_2))


# In[25]:


print(len(ano_obs))


# In[26]:


ano_features = ano_obs.iloc[0:164499, :]
columns = ['id','attack_cat','label']
ano_features = ano_obs.drop(columns, axis = 1)
ano_features


# In[27]:


x_test = normalize(ano_features)
x_test["is_sm_ips_ports"].fillna("0.0", inplace = True) 
x_test


# In[28]:


y_test = ano_obs.iloc[:,-1]
y_test


# In[29]:


y_test_final = y_train_2.append(y_test)
y_test_final


# In[30]:


x_train_1


# In[31]:


y_train_1


# In[32]:


x_test_final = x_train_2.append(x_test)
x_test_final


# In[33]:


y_test_final


# In[34]:


from sklearn import svm


# In[35]:


oneclass = svm.OneClassSVM(kernel='rbf', gamma=0.01, nu=0.95)
oneclass.fit(x_train_1)


# In[36]:


root = Tk()



def Run_python():
    os.system('python graph.py')
    
root.title("Intrusion Detection System Using Machine Learning")

label_0 = Label(root, text="Intrusion Detection System",width=20,font=("bold", 20))
label_0.grid(row = 0 , column = 1)


label_1 = Label(root, text="Upload the csv file here",width=20,font=("bold", 10))
label_1.grid(row = 1 , column = 1)

def import_csv_data():
    global v
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)
    df = pd.read_csv(csv_file_path)
    print(df.head())



tk.Label(root, text='File Path').grid(row=2, column=1)
v = tk.StringVar()
entry = tk.Entry(root, textvariable=v).grid(row=2, column=2)
tk.Button(root, text='Browse Data Set',command=import_csv_data).grid(row=2, column=3)
tk.Button(root, text='Upload',command=root.destroy).grid(row=3, column=2)


#tk.Button(root,text="Display graphical data",command=Run_python).grid(row=4,column=1)




root.mainloop()


# In[37]:


fraud_pred = oneclass.predict(x_test_final)


# In[38]:


fraud_pred


# In[39]:


y_test_final


# In[40]:


from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test_final, fraud_pred)
acc_percentage = acc_score * 100
acc_percentage


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test_final,fraud_pred))
print(classification_report(y_test_final,fraud_pred))


# In[42]:


from win10toast import ToastNotifier
toaster = ToastNotifier()
toaster.show_toast("IDS Alert","File processing complete, check results now")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




