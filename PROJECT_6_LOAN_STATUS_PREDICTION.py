#!/usr/bin/env python
# coding: utf-8

# # IMPORTING IMPORTANT LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score , recall_score , precision_score , f1_score , confusion_matrix ,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# importing dataset
data=pd.read_csv('Loan Prediction Dataset.csv')


# In[3]:


data


# In[4]:


data.isnull().sum()


# In[5]:


data.describe()


# In[6]:


data.info()


# # HANDLING NULL VALUES
# 

# In[17]:


data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].mean())


# In[8]:


data['Loan_Amount_Term']=data['Loan_Amount_Term'].replace(np.nan,data['Loan_Amount_Term'].mean())


# In[9]:


data['LoanAmount']=data['LoanAmount'].replace(np.nan,data['LoanAmount'].mean())


# In[10]:


data['Self_Employed']=data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])


# In[11]:


data['Dependents']=data['Dependents'].fillna(data['Dependents'].mode()[0])


# In[12]:


data['Married']=data['Married'].fillna(data['Married'].mode()[0])


# In[13]:


data['Gender']=data['Gender'].fillna(data['Gender'].mode()[0])


# In[18]:


data.isnull().sum()


# # EDA

# In[19]:


sns.countplot(data['Gender'])
plt.show()


# In[20]:


sns.countplot(data['Married'])
plt.show()


# In[21]:


sns.countplot(data['Dependents'])
plt.show()


# In[22]:


sns.countplot(data['Education'])
plt.show()


# In[23]:


sns.countplot(data['Self_Employed'])
plt.show()


# In[24]:


sns.displot(data['ApplicantIncome'])
plt.show()


# In[25]:


sns.displot(data['CoapplicantIncome'])
plt.show()


# In[26]:


sns.displot(data['LoanAmount'])
plt.show()


# In[27]:


sns.displot(data['Loan_Amount_Term'])
plt.show()


# In[28]:


sns.countplot(data['Loan_Amount_Term'])
plt.show()


# In[35]:


sns.displot(data['Credit_History'])
plt.show()


# # CREATING NEW VARIABLE

# In[36]:


data['Total_Income']=data['ApplicantIncome']+data['CoapplicantIncome']


# # LOG TRANSFORMATION

# In[38]:


data.columns


# In[39]:


data['Total_Income']=np.log(data['Total_Income']+1)


# In[40]:


data['ApplicantIncome']=np.log(data['ApplicantIncome']+1)


# In[41]:


data['CoapplicantIncome']=np.log(data['CoapplicantIncome']+1)


# In[42]:


data['LoanAmount']=np.log(data['LoanAmount']+1)


# In[43]:


data['Loan_Amount_Term']=np.log(data['Loan_Amount_Term']+1)


# In[45]:


sns.displot(data['Total_Income'])
plt.show()


# In[46]:


sns.displot(data['ApplicantIncome'])
plt.show()


# In[47]:


sns.displot(data['CoapplicantIncome'])
plt.show()


# In[48]:


sns.displot(data['LoanAmount'])
plt.show()


# In[49]:


sns.displot(data['Loan_Amount_Term'])
plt.show()


# In[50]:


data


# # CORRELATION MATRX

# In[51]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# # PREPROCESSING

# In[55]:


LE=LabelEncoder()
cols=['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History', 'Property_Area', 'Loan_Status']
for i in cols:
    LE.fit(data[i])
    data[i]=LE.transform(data[i])


# In[64]:


data=data.drop(['Loan_ID'],axis=1)


# # SPLITTING DATA INTO FEATURES AND TARGETS

# In[65]:


data.columns


# In[66]:


X=data[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area',
       'Total_Income']]


# In[67]:


y=data['Loan_Status']


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# # MODEL BUILDING AND TRAINING

# In[69]:


def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test)*100)
    # cross validation - it is used for better validation of model
    # eg: cv-5, train-4, test-1
    score = cross_val_score(model, x, y, cv=5)
    print("Cross validation is",np.mean(score)*100)


# In[70]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# In[71]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
model = RandomForestClassifier()
classify(model, X, y)


# In[72]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model, X, y)


# In[73]:


model = ExtraTreesClassifier()
classify(model, X, y)


# In[ ]:




