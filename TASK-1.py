#!/usr/bin/env python
# coding: utf-8

# # TASK-1 PREDICTION USING SUPERVISED ML (simple linear regression)
# 

# **- by sindhura gundubogula** 

# In this task we predict percentage of marks of a student based on number of study hours using linear regression as we have only two variables.
# 
# Data is available at the URL: http://bit.ly/w-data

# **STEP-1 import all required python libraries**

# In[147]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# **STEP-2 Load the dataset**

# In[148]:


raw_data = pd.read_csv("http://bit.ly/w-data")


# In[149]:


raw_data


# In[150]:


raw_data.info()


# **STEP-3 EXPLORATORY DATA ANALYSIS**
# 
#    our dataset has two columns HOURS and SCORES with no missing and inconsistent data.We don't have any categorical data as both the columns include only numerical data of type Float and int. Hence, No data cleaning is required for our dataset

# **Exploring descriptive statistics and derterming vairaibles of interest**

# In[151]:


raw_data.describe(include='all')


# In[152]:


sns.distplot(raw_data['Hours'])


# In[153]:


sns.distplot(raw_data['Scores'])


# No outliers are found

# **STEP-4 LINEAR REGRESSION**

# **Declare inputs and outputs**

# In[262]:


#y = raw_data['Scores']
#x = raw_data['Hours']

x = raw_data.iloc[:, :-1].values  
y = raw_data.iloc[:, 1].values  


# In[263]:


np.reshape(y, (25, ))


# In[264]:


np.reshape(x, (25, ))


# **Exploring the data**

# In[265]:


raw_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Score')  
plt.xlabel('study hours')  
plt.ylabel('score')  
plt.show()


# From the graph above, we can clearly see that the score increases as the number of hours increases i.e, a positive linear relation 

# **Split TRAIN and TEST data**

# In[266]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# **Training the model**

# In[267]:


reg = LinearRegression()

reg.fit(x_train,y_train)
print("training the model is successful")


# **estimating training prediction**

# In[268]:


y_hat = reg.predict(x_train) #predictions are stored in y^(y_train should match y_hat)


# In[269]:




plt.scatter(y_train,y_hat)
plt.title('y_train vs y_hat')  
plt.xlabel('y_train',size=18)  
plt.ylabel('y_hat',size=18) 
plt.show()


# In[270]:


sns.distplot(y_train-y_hat)
plt.title("residual PDF")


# In[271]:


reg.score(x_train,y_train)


# **finding weights and bias(intercept and coeff)**

# In[272]:


reg.intercept_


# In[273]:


reg.coef_


# **testing the model**

# In[274]:


y_hat_test = reg.predict(x_test)


# In[275]:



plt.scatter(y_test,y_hat_test)
plt.title('y_test vs y_hat_test')  
plt.xlabel('y_test',size=18)  
plt.ylabel('y_hat_train',size=18) 

plt.show()


# **Comparing actual and predicted values**

# In[278]:


plt.scatter(x_test,y_test, color = 'grey')
plt.scatter(x_test,y_hat_test)


plt.show()


# In[279]:


plt.scatter(x_test,y_test)
plt.plot(x_test,y_hat_test,color = 'green')


plt.show()


# In[230]:


# Comparing Actual vs Predicted
df_performance = pd.DataFrame({'Actual': y_test, 'Predicted': y_hat_test})  
df_performance


# In[231]:


#calculating residuals
df_performance['residuals']=df_performance['Actual']-df_performance['Predicted']
df_performance


# In[232]:


#difference%
df_performance['difference%']=np.absolute(df_performance['residuals']/df_performance['Actual']*100)
df_performance


# In[233]:


df_performance.describe()


# **testing with own data**
# 
# **what will be the score if student studies for 9.25 hrs/day?**

# In[338]:


hours = 9.25
score_predicted = reg.predict(np.array(hours).reshape(1,1))
print(' \033[1;30m Answer: The score of the student if he studies for {}hrs/day is \033[1;32m {} '.format(hours,*score_predicted))


# **STEP:5 Evaluating the model**

# We can choose any of the available metrics, here we choose **mean square error**. This step is a concluding step of our regression process and is important to compare various algorithms performed on the same dataset

# In[339]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_hat_test)) 


#  ** <<**end of the algorithm**>> **
