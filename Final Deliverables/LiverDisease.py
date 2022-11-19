# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[2]:


data = pd.read_csv(r'C:\Users\Hema\Desktop\IBMProject\indian_liver_patient (1).csv')


# In[3]:


data.info()


# In[4]:


data.head(10)


# In[5]:


data.tail(10)


# In[6]:


data.describe()


# In[7]:


data.isnull().any()


# In[8]:


data.isnull().sum()


# In[9]:


data['Albumin_and_Globulin_Ratio']=data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mode()[0])


# In[10]:


data.isnull().sum()


# In[11]:


sns.countplot(data=data,x='Gender',label='Count')
m,f=data['Gender'].value_counts()
print("No of Males:",m)
print("no of Females:",f)


# In[12]:


sns.countplot(data=data,x='Dataset')
LD,NLD=data['Dataset'].value_counts()
print("liver disease patients:",LD)
print("non-liver disease patients:",NLD)


# In[13]:


def partition(x):
    if x=='Male':
        return 1
    return 0
data['Gender']=data['Gender'].map(partition)


# In[14]:


#data


# In[15]:


def partition(x):
    if x==2:
        return 0
    return 1

data['Dataset']=data['Dataset'].map(partition)

# In[16]:


data['Dataset']


# In[17]:


x=data.iloc[:,0:-1].values
y=data.iloc[:,-1].values


# In[18]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=42)


# In[19]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier=KNeighborsClassifier(n_neighbors=21,metric='minkowski')
knn_classifier.fit(xtrain,ytrain)


# In[21]:


knn_y_pred=knn_classifier.predict(xtest)


# In[22]:


from sklearn.svm import SVC
svm_classifier=SVC(kernel='rbf',random_state=0)
svm_classifier.fit(xtrain,ytrain)


# In[23]:


svm_y_pred=svm_classifier.predict(xtest)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
RFModel=RandomForestClassifier()
RFModel.fit(xtrain,ytrain)


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


SVMaccuracy=accuracy_score(svm_y_pred,ytest)
SVMaccuracy


# In[31]:


RFpred=RFModel.predict(xtest)


# In[32]:


RFaccuracy=accuracy_score(RFpred,ytest)
RFaccuracy


# In[36]:


from sklearn.metrics import confusion_matrix


# In[37]:


RFcm=confusion_matrix(RFpred,ytest)
RFcm


# In[38]:


KNNaccuracy=accuracy_score(knn_y_pred,ytest)
KNNaccuracy


# In[39]:


KNNcm=confusion_matrix(knn_y_pred,ytest)
KNNcm


# In[40]:


import pickle
pickle.dump(knn_classifier,open('liver_analysis.pkl','wb'))


# In[ ]:
