#!/usr/bin/env python
# coding: utf-8

# In[2]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[3]:
claimsds = pd.read_csv(r"C:\Users\dell\Documents\Data Science\Excel\Capstone02\Auto_Insurance_Claims_MasterData.csv", header =0)


# In[4]:
claimsds.head()


# In[5]:


claimsds.info()


# In[6]:


claims = claimsds.drop(columns =['Inception_dt','Policy_No', 'Accident.date', 'Police_File', 'Any_Eye_Witness', 'Hired_Attorney', 'Claim_Number', 'Claim_Paid_Out', 'Claim_Type_BI', 'Make', 'Claimed_Target2',])


# In[7]:


from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

claims['Policy_Type'] = number.fit_transform(claims['Policy_Type'],astype='int')
claims['Policy_Premium_Range'] = number.fit_transform(claims['Policy_Premium_Range'])
claims['Channel'] = number.fit_transform(claims['Channel'])
claims['State'] = number.fit_transform(claims['State'])
claims['Age_Range'] = number.fit_transform(claims['Age_Range'])
claims['Gender'] = number.fit_transform(claims['Gender'])
claims['Marital_Status'] = number.fit_transform(claims['Marital_Status'])
claims['Education'] = number.fit_transform(claims['Education'])
claims['Profession'] = number.fit_transform(claims['Profession'])
claims['Vehicle_Usage'] = number.fit_transform(claims['Vehicle_Usage'])
claims['Coverage_Type'] = number.fit_transform(claims['Coverage_Type'])
claims['Umbrella_Policy'] = number.fit_transform(claims['Umbrella_Policy'])
claims['Vehicle_Cost_Range'] = number.fit_transform(claims['Vehicle_Cost_Range'])
claims['Road_Type'] = number.fit_transform(claims['Road_Type'])
claims['Accident_Severity'] = number.fit_transform(claims['Accident_Severity'])
claims['Driving_Exp_Range'] = number.fit_transform(claims['Driving_Exp_Range'])
claims['Annual_Miles_Range'] = number.fit_transform(claims['Annual_Miles_Range'])


# In[8]:


claims.info()


# In[9]:


#load data into dependent and independent variables

IndepVar = []

for col in claims.columns:
    if col != 'Claimed_Target1':
        IndepVar.append(col)
        
TargetVar = ['Claimed_Target1']

X = claims[IndepVar]
y = claims[TargetVar]


# In[10]:


X


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# # LogisticRegression

# In[12]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# In[13]:


#Build the model

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[14]:


y_pred = logreg.predict(X_test)


# In[28]:


params = logreg.get_params()
print(params)


# In[15]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))


# In[16]:


#import scikit_learn metrics module for accuracy calculation
from sklearn import metrics

#Moddel Accuracy: how often is the classifier correct?
print('Accuracy:', metrics.accuracy_score(y_test,y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test,y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test,y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("f1-score:",metrics.f1_score(y_test,y_pred))


# # RandomForestClassifier

# In[17]:


from sklearn.ensemble import RandomForestClassifier


# In[18]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[19]:


y_pred1 = rfc.predict(X_test)


# In[20]:


params = rfc.get_params()
print(params)


# In[21]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))
print(accuracy_score(y_test, y_pred1)*100, '%')


# In[22]:


#import scikit_learn metrics module for accuracy calculation
from sklearn import metrics

#Moddel Accuracy: how often is the classifier correct?
print('Accuracy:', metrics.accuracy_score(y_test,y_pred1))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test,y_pred1))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test,y_pred1))

# Model Recall: what percentage of positive tuples are labelled as such?
print("f1-score:",metrics.f1_score(y_test,y_pred1))


# # KNN

# In[23]:


from sklearn.preprocessing import StandardScaler 

ss = StandardScaler()
ss.fit(X)
ss_transform = ss.transform(X)
sc_claims = pd.DataFrame(ss_transform)

sc_claims.head()


# In[24]:


from sklearn.model_selection import train_test_split

X = ss_transform
y = np.ravel(claims[TargetVar])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

accuracy = []

for a in range (1, 5, 1):
    k = a
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    y_preds2 = knn.predict(X_test)
    ascore = accuracy_score(y_test, y_preds2)*100
    ascore = "{:.2f}".format(ascore)
    print('Accuracy value for k=', k, 'is', ascore)


# In[26]:


print(confusion_matrix(y_test, y_preds2))
print(classification_report(y_test, y_preds2))


# In[27]:


#import scikit_learn metrics module for accuracy calculation
from sklearn import metrics

#Moddel Accuracy: how often is the classifier correct?
print('Accuracy:', metrics.accuracy_score(y_test,y_preds2))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test,y_preds2))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test,y_preds2))

# Model Recall: what percentage of positive tuples are labelled as such?
print("f1-score:",metrics.f1_score(y_test,y_preds2))


# In[ ]:




