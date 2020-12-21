#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# Regression is a form of Predictive Modelling technique which investigates the relationship between a dependent and independent variable.

# In simple words, a Regression analysis involves graphing a line over a set of datapoints that most closely fits the overall shape of the data.
# 
# A Regression shows the changes in a dependent variable on the y-axis to the changes in the explanatory variable on the x-axis

# Now, what are the Uses of Regression Analysis:- There are 3 major uses of regression analysis
# 1) Determining the strength of predicators 
# 
#     Ex:- Relationship between sales and marketing spending
# 
#          Relationship between age and income
# 
# 2) Forecasting an effect
# 
#     Ex:- how much additional sale income do we get for each 1000 dollars i spend on marketing
# 
# 3) Trend Forecasting
# 
#     Ex:- Price of Bitcoin in next 6 months
# 

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[2]:


# Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.isnull().sum()


# In[6]:


data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# Custom Methods

# In[7]:


#Declaring these for comparision between models

#predictions=pd.DataFrame()
MeanAbsoluteError_dict={}
MeanSquaredError_dict={}
RootMeanSquaredError_dict={}
R2Score_dict={}
TestR2Score_dict={}
CrossValScore_dict={}


# In[8]:


def consolidator(modelInstance,ModelName,X_train,Y_train,X_test,Y_test,Y_Pred):
  #predictions[ModelName] = Y_Pred

    print(ModelName,'Model:-')

    R2Score = modelInstance.score(X_train, Y_train)
    TestR2Score = modelInstance.score(X_test, Y_test)
    MeanAbsErr = mean_absolute_error(Y_test, Y_Pred)
    MeanSqErr =  mean_squared_error(Y_test, Y_Pred)
    RootMeanSqErr = np.sqrt(mean_squared_error(Y_test, Y_Pred))
  
    print("R^2 Score: ", R2Score)
    print("Test R^2 Score: ", TestR2Score)
    print('Mean Absolute Error :', MeanAbsErr)  
    print('Mean Squared Error :', MeanSqErr)  
    print('Root Mean Squared Error :', RootMeanSqErr)

    R2Score_dict[ModelName] = R2Score
    TestR2Score_dict[ModelName] = TestR2Score
    MeanAbsoluteError_dict[ModelName] =  MeanAbsErr
    MeanSquaredError_dict[ModelName] = MeanSqErr
    RootMeanSquaredError_dict[ModelName] = RootMeanSqErr


# # *Training:-*

# In[9]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[11]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[12]:


LR = LinearRegression()
param_grid = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

grid_search = GridSearchCV(estimator = LR, param_grid = param_grid,cv=3,verbose=2)
LR_GS = grid_search.fit(X_train, y_train)


# In[13]:


LR_GS.best_estimator_


# Here, we got the best estimator as - LinearRegression(normalize=True)

# In[14]:


regressor = LinearRegression(normalize=True)
regressor.fit(X_train, y_train) 


# In[15]:


Y_pred_LR = regressor.predict(X_test)


# # *Model Evaluation*

# In[16]:


consolidator(regressor,'LR_Original',X_train,y_train,X_test,y_test,Y_pred_LR)


# In[17]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.show()


# # *Predicting Test Data*

# In[18]:


y_pred = regressor.predict(X_test) # Predicting the scores


# In[19]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # *Predicting score at 9.25 hours*

# In[20]:


pred = regressor.predict([[9.25]])


# In[21]:


pred


# In[22]:


print('Predicted Score = ',pred[0])


# In[ ]:




