#!/usr/bin/env python
# coding: utf-8

# # Importing Required Libaries:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Read the Data:

# In[2]:


data=pd.read_csv(r"E:\internship\Dataset\train.csv")


# # Data description:

# In[3]:


data.head(2)


# In[4]:


data.tail(2)


# In[5]:


#data.drop(["Unnamed: 0"],axis=1,inplace=True)


# In[6]:


data.head(2)


# In[7]:


print(data.shape)
print("No.of rows in the dataset:",data.shape[0])
print("No.of columns in the dataset:",data.shape[1])


# In[8]:


data.info()


# In[9]:


data.describe()


# In[10]:


data.columns


# In[11]:


Numerical_value=[feature for feature in data.columns if data[feature].dtype!="O"]
categorical_value=[feature for feature in data.columns if data[feature].dtype=="O"]


# In[12]:


print("We have {} numrerical values in dataset".format(len(Numerical_value)))
print("We have {} categorical values in dataset".format(len(categorical_value)))


# # Data Preprocessing:

# # 1.Convert Categorical values into Numerical values:
# # 2.Check the null values in the dataset:

# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


labelEncoder=LabelEncoder()

data["Item_Identifier"]=labelEncoder.fit_transform(data["Item_Identifier"])
data["Item_Fat_Content"]=labelEncoder.fit_transform(data["Item_Fat_Content"])
data["Item_Type"]=labelEncoder.fit_transform(data["Item_Type"])
data["Outlet_Identifier"]=labelEncoder.fit_transform(data["Outlet_Identifier"])
data["Outlet_Size"]=labelEncoder.fit_transform(data["Outlet_Size"])
data["Outlet_Location_Type"]=labelEncoder.fit_transform(data["Outlet_Location_Type"])
data["Outlet_Type"]=labelEncoder.fit_transform(data["Outlet_Type"])


# In[15]:


data.head(5)


# In[16]:


data.info()


# In[17]:


data.isnull().sum()


# # Filling Null values with Simple Imputation:

# In[18]:


from sklearn.impute import SimpleImputer


# In[19]:


Imputation=SimpleImputer(strategy='mean')

Imputed_data=Imputation.fit_transform(data)

print(Imputed_data)


# In[20]:


Final_data=pd.DataFrame(Imputed_data,columns=data.columns)

Final_data


# In[21]:


Final_data.isnull().sum()


# # EDA (Expolartary Data Analysics):   KDE= Kernel Distrubution Function

# In[22]:


sns.histplot(data=data,x=data["Item_Outlet_Sales"],bins=50,kde="True",color="g")
plt.show()


# In[23]:


Final_data.hist(bins=50,figsize=(20,20))
plt.show()


# In[24]:


plt.plot(data["Outlet_Establishment_Year"],data["Item_Outlet_Sales"])
plt.show()


# # Correlation:

# In[25]:


Final_data.corr()


# In[26]:


fig,ax=plt.subplots(figsize=(20,10))
sns.heatmap(Final_data.corr(),annot=True)
plt.show()


# # Splitting the data into Two features ( x -- Dependent & y -- Independent features):

# In[27]:


x=Final_data.drop(["Item_Outlet_Sales"],axis=1)
x.head(2)


# In[28]:


y=Final_data["Item_Outlet_Sales"]
y


# # Feature Selection Method: (RFE: Recursive Feature Elimination)

# In[29]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[30]:


LR=LinearRegression()
rfe=RFE(LR,n_features_to_select=8)

rfe.fit(x,y)

selected_features=x.columns[rfe.support_]

print(selected_features)


# In[31]:


X=Final_data.drop(["Item_Weight","Item_Type","Item_Identifier","Item_Outlet_Sales"],axis=1)
X


# In[32]:


y


# # Splitting the Dataset into Training and Testing Data: ( Training_data=80% , Testing_data=20%)

# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[35]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Model Training:

# In[36]:


LR=LinearRegression()
LR.fit(X_train,y_train)
LR.score(X_test,y_test)


# In[37]:


y_pred_LR=LR.predict(X_test)


# In[38]:


from sklearn.metrics import r2_score,mean_squared_error


# In[39]:


print("MSE:",mean_squared_error(y_test,y_pred_LR))
print("R-squared value:",r2_score(y_test,y_pred_LR))


# In[40]:


from sklearn.linear_model import Ridge


# In[41]:


RG=Ridge()
RG.fit(X_train,y_train)
RG.score(X_test,y_test)


# In[42]:


y_pred_RG=RG.predict(X_test)


# In[43]:


print("MSE:",mean_squared_error(y_test,y_pred_RG))
print("R-squared value:",r2_score(y_test,y_pred_RG))


# In[44]:


from sklearn.linear_model import Lasso


# In[45]:


LS=Lasso()
LS.fit(X_train,y_train)
LS.score(X_test,y_test)


# In[46]:


y_pred_LS=LS.predict(X_test)


# In[47]:


print("MSE:",mean_squared_error(y_test,y_pred_LS))
print("R-squared value:",r2_score(y_test,y_pred_LS))


# In[48]:


from sklearn.tree import DecisionTreeRegressor


# In[49]:


DT=DecisionTreeRegressor()
DT.fit(X_train,y_train)
DT.score(X_test,y_test)


# In[50]:


y_pred_DT=DT.predict(X_test)


# In[51]:


print("MSE:",mean_squared_error(y_test,y_pred_DT))
print("R-squared value:",r2_score(y_test,y_pred_DT))


# In[52]:


from sklearn.ensemble import RandomForestRegressor


# In[53]:


RF=RandomForestRegressor()
RF.fit(X_train,y_train)
RF.score(X_test,y_test)


# In[54]:


y_pred_RF=RF.predict(X_test)


# In[55]:


print("MSE:",mean_squared_error(y_test,y_pred_RF))
print("R-squared value:",r2_score(y_test,y_pred_RF))


# In[56]:


from sklearn.svm import SVR


# In[57]:


SV=SVR(kernel='linear')
SV.fit(X_train,y_train)
SV.score(X_test,y_test)


# In[58]:


y_pred_SV=SV.predict(X_test)


# In[59]:


print("MSE:",mean_squared_error(y_test,y_pred_SV))
print("R-squared value:",r2_score(y_test,y_pred_SV))


# In[60]:


from sklearn.ensemble import GradientBoostingRegressor


# In[61]:


GBR=GradientBoostingRegressor()
GBR.fit(X_train,y_train)
GBR.score(X_test,y_test)


# In[62]:


y_pred_GBR=GBR.predict(X_test)


# In[63]:


print("MSE:",mean_squared_error(y_test,y_pred_GBR))
print("R-squared value:",r2_score(y_test,y_pred_GBR))


# In[64]:


from xgboost import XGBRegressor


# In[65]:


XGB=XGBRegressor()
XGB.fit(X_train,y_train)
XGB.score(X_test,y_test)


# In[66]:


y_pred_XGB=XGB.predict(X_test)


# In[67]:


print("MSE:",mean_squared_error(y_test,y_pred_XGB))
print("R-squared value:",r2_score(y_test,y_pred_XGB))


# # Hyperparameter Tuning:

# In[68]:


param_grid={
    'max_depth': [100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [200, 300]
    }


# In[69]:


RF_HY=RandomForestRegressor()


# In[70]:


from sklearn.model_selection import GridSearchCV

grid_search=GridSearchCV(estimator=RF_HY,cv=3,param_grid=param_grid)
grid_search.fit(X_train,y_train)


# In[71]:


grid_search.best_params_


# In[74]:


RF_HYP=RandomForestRegressor(max_depth= 110,
 max_features= 3,
 min_samples_leaf= 5,
 min_samples_split=12,
 n_estimators= 300)
RF_HYP.fit(X_train,y_train)
RF_HYP.score(X_test,y_test)


# In[76]:


y_pred_RF_HYP=RF_HYP.predict(X_test)


# In[77]:


print("MSE:",mean_squared_error(y_test,y_pred_RF_HYP))
print("R-squared value:",r2_score(y_test,y_pred_RF_HYP))


# In[78]:


param_grid={
    'learning_rate': [0.01,0.001],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [8, 10],
    'n_estimators': [200, 300]
    }


# In[79]:


GBR_HY=GradientBoostingRegressor()


# In[80]:


grid_search=GridSearchCV(estimator=GBR_HY,cv=3,param_grid=param_grid)
grid_search.fit(X_train,y_train)


# In[81]:


grid_search.best_params_


# In[82]:


GBR_HYP=GradientBoostingRegressor()
GBR_HYP.fit(X_train,y_train)
GBR.score(X_test,y_test)


# In[83]:


y_pred_GBR_HYP=GBR_HYP.predict(X_test)


# In[84]:


print("MSE:",mean_squared_error(y_test,y_pred_GBR_HYP))
print("R-squared value:",r2_score(y_test,y_pred_GBR_HYP))


# In[85]:


import pickle


# In[86]:


with open(r"E:\internship\Dataset/RF.pkl", "wb") as file:
    pickle.dump(RF_HYP, file)


# In[87]:


X.columns


# In[88]:


Item_Fat_Content=float(input("Enter the value:  "))
Item_Visibility=float(input("Enter the value:  "))
Item_MRP=float(input("Enter the value:"))
Outlet_Identifier=float(input("Enter the value:  "))
Outlet_Establishment_Year=int(input("Enter the value:  "))
Outlet_Size=float(input("Enter the value:  "))
Outlet_Location_Type=float(input("Enter the value:  "))
Outlet_Type=float(input("Enter the value:   "))


unknown_value=[Item_Fat_Content, Item_Visibility, Item_MRP, Outlet_Identifier,
       Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type,Outlet_Type]


# In[89]:


# load the model:
loaded_model = pickle.load(open(r"E:\internship\Dataset\RF.pkl", 'rb'))


# In[91]:


y_predict=loaded_model.predict([unknown_value])


# In[92]:


y_predict[0]


# In[94]:


print("predicted Sales Amount is RS:",y_predict[0])


# In[ ]:




