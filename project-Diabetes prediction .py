#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[7]:


#data collection and anaysis
diabetes_dataset=pd.read_csv('diabetes (1).csv')


# In[8]:


diabetes_dataset.head()


# In[9]:


diabetes_dataset.describe()


# In[14]:


diabetes_dataset['Outcome'].value_counts()
#0 means non diabetic,1 means diabetic


# In[15]:


diabetes_dataset.groupby('Outcome').mean()


# In[20]:


#separating the data and labels
x=diabetes_dataset.drop(columns= 'Outcome', axis=1)
y=diabetes_dataset['Outcome']


# In[21]:


print(x)


# In[22]:


print(y)


# In[23]:


#data standardization
scaler= StandardScaler()


# In[26]:


scaler.fit(x)


# In[27]:


standardized_data=scaler.transform(x)


# In[28]:


print(standardized_data)


# In[29]:


x=standardized_data


# In[30]:


print(x)
print(y)


# In[31]:


#train test split
x_train, x_test,y_train, y_test= train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)


# In[32]:


#training the model
classifier = svm.SVC(kernel='linear')


# In[34]:


#training the support vector machine classifier
classifier.fit(x_train,y_train)


# In[35]:


#model evaluation
#accuracy score of training data
x_train_prediction= classifier.predict(x_train)
taining_data_accuracy= accuracy_score(x_train_prediction,y_train)


# In[37]:


print('accuracy score of training data:',taining_data_accuracy)


# In[38]:


#accuracy score of test data
x_test_prediction= classifier.predict(x_test)
test_data_accuracy= accuracy_score(x_test_prediction,y_test)


# In[39]:


print('accuracy score of test data:',test_data_accuracy)


# In[41]:


input_data= (1,89,66,23,94,28.1,0.167,21)
#changing the input data in nparray
input_datanp= np.asarray(input_data)
#reshape the array as we are predicting for one instance
input_data_reshape =input_datanp.reshape(1,-1)
#standarize the input data
std_data= scaler.transform(input_data_reshape)
print(std_data)
prediction =classifier.predict(std_data)
print(prediction)
if (prediction == 0):
    print('not diabetic')
else:
    print('diabetic')


# In[ ]:




