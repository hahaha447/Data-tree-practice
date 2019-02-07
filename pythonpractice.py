#!/usr/bin/env python
# coding: utf-8

# # #Notebook for decision tree---Tiancheng

# In[11]:


import pandas as pd
import os
os.chdir("C:/Users/Arche/graduate/R working directory/5505/project")# Set my working directory 
os.getcwd()

prac=pd.read_csv("prac.csv")#Read data
prac=prac.dropna(axis=0)
prac.head()


# In[2]:


from sklearn import tree
model=tree.DecisionTreeClassifier()
input= prac.drop(["Survived","Name","Ticket","SibSp","Cabin","Embarked"],axis=1) # Get our input data
target=prac["Survived"] # Get the result data(the value we want to perdict)
input.head()


# In[10]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
input["sex_new"]=le.fit_transform(prac["Sex"])# Transfor our label from string type to integer type
input_n=input.drop(["Sex"],axis=1)
input_n.head()


# In[5]:


model.fit(input_n,target) # fit the model


# In[6]:


model.score(input_n,target)


# In[1]:


model.predict([[3,20,0,50,1]])# useing the model for perdict


# In[ ]:




