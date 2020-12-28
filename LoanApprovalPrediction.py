#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np        # Fundamental package for linear algebra and multidimensional arrays
import pandas as pd       # Data analysis and manipultion tool


# In[3]:


##sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


# In[4]:


# In read_csv() function, we have passed the location to where the files are located in the dphi official github page.
#auction_data = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/auction_data/train_set_label.csv" )
#loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv")
#insurance_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/travel_insurance/Training_set_label.csv" )


# In[5]:


#insurance_data.to_csv("insurance_train_data.csv", index=False,encoding='utf-8', header=True)
#loan_data.to_csv("loan_data.csv", index=False,encoding='utf-8', header=True)


# In[6]:


loan_data=pd.read_csv("loan_data.csv")


# In[7]:


loan_data.head(5)


# In[8]:


loan_data.isna().sum()


# In[9]:


loan_data.shape


# In[10]:


#insurance_data['Gender']=insurance_data['Gender'].fillna("Not Specified") 


# In[20]:


#ax=insurance_data['Claim'].plot(kind='bar',figsize=(10,6))
#plot.ax

loan_data['Self_Employed'].value_counts()


# In[11]:


#insurance_data.dropna(inplace=True)
#insurance_data.shape


# In[12]:


loan_data.columns


# In[13]:


loan_data.columns = loan_data.columns.str.strip()
loan_data.columns


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(loan_data.drop('Loan_Status', axis=1), loan_data.Loan_Status,
                                                   test_size=0.2, random_state=0)


# In[15]:


y_train.isna().sum()


# In[16]:


num_cols = [col for col in X_train.columns if X_train[col].dtypes!='O']
num_cols


# In[17]:


cat_cols = [col for col in X_train.columns if X_train[col].dtypes=='O']
cat_cols


# In[18]:


pp_num = Pipeline([
    ('num_imp', SimpleImputer(strategy='median', add_indicator=False)),
    ('rob_num', RobustScaler())
])

pp_cat = Pipeline([
    ('cat_imp', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('ohe_cat', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])


# In[19]:


from sklearn.impute import MissingIndicator


# In[20]:


ct = ColumnTransformer([
    ('mi', MissingIndicator(), X_train.columns),
    ('pp_num', pp_num, num_cols),
    ('pp_cat', pp_cat, cat_cols)
])


# In[21]:


xt = ct.fit_transform(X_train)
xt


# In[22]:


pd.DataFrame(xt).head()


# In[23]:


pd.DataFrame(xt).isna().sum().sum()


# In[24]:


pipe_final = Pipeline([
    ('ct_step', ct),
    ('model', DecisionTreeClassifier())
])


# In[25]:


pipe_final.fit(X_train, y_train)


# In[26]:


X_test.head()


# In[27]:


ct.transform(X_test)


# In[28]:


X_test.isna().sum()


# In[29]:


pipe_final.predict(X_test)


# In[30]:


pipe_final.score(X_test, y_test)


# In[51]:


X_test.columns


# In[31]:


new_vals= pd.DataFrame([['LP001116','Male','No','0','Not Graduate','No',3748,1668.0,110.0,360.0,1.0,'Semiurban']],columns = X_test.columns)
new_pred= pipe_final.predict(new_vals)
#result = pd.DataFrame({"X=%s, Predicted=%s" % (new_vals[0], new_pred[0])})


# In[32]:


print(new_pred)


# In[33]:


pipe_final.named_steps


# In[34]:


pipe_final.named_steps['ct_step']


# In[35]:


model=pipe_final.named_steps['model']


# In[36]:


#X_val=pd.DataFrame('LP001116','Male','No',0,'Not Graduate','No',3748,1668.0,110.0,360.0,1.0,'Semiurban')
pipe_final.predict_proba(new_vals)


# In[48]:


import pickle
filename="LoanApprovalPrediction_PipeFinal.pkl"
pickle.dump(pipe_final,open(filename,'wb'))


# In[50]:


LoanModel_ppl = pickle.load(open('LoanApprovalPrediction_PipeFinal.pkl','rb'))

print(LoanModel_ppl.predict(new_vals))


# In[45]:


#test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/auction_data/test_set_label.csv')
#test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/travel_insurance/Testing_set_label.csv')
test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')


# In[ ]:


test_data.head()


# In[ ]:


test_data.isna().sum()


# In[ ]:


test_data.shape


# In[ ]:


ct.transform(test_data)


# In[ ]:


target = pipe_final.predict(test_data)


# In[ ]:


target


# In[ ]:





# In[ ]:


res = pd.DataFrame(target) #target is nothing but the final predictions of your model on input features of your new unseen test data

res.index = test_data.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]
res.to_csv("submission_loan.csv",index=False) 


# In[ ]:


res['prediction'].unique()


# In[ ]:


import shap


# In[ ]:


def model_predict(data_asarray):
    data_asframe =  pd.DataFrame(data_asarray, columns=X_test.columns)
    return pipe_final.predict_proba(data_asframe)


# In[ ]:


explainer = shap.TreeExplainer(pipe_final.named_steps['model'])
#shap.LinearExplainer(logmodel)
shap_values = explainer.shap_values(xt)


# In[ ]:


shap.summary_plot(shap_values, xt, plot_type='bar')


# In[ ]:



explainer.expected_value


# In[ ]:




