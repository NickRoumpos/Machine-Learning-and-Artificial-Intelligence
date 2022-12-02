#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import plot_confusion_matrix
## Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="dark")


# # Data Pre-**Processing**

# # Load Data
# 
# 

# In[2]:


import pandas as pd
df=pd.read_csv("/Users/nikolaosroumpos/Desktop/Dataset4.csv",sep=";")
df


# In[3]:


# Drop 'duration' column
df_bank = df.drop('duration', axis=1)

# print(df_bank.info())
print('Shape of dataframe:', df_bank.shape)
df_bank.head()


# ## Class Distribution
# 

# In[4]:


df_bank['y'].value_counts()


# In[5]:


sns.countplot(x="y", data=df_bank)


# As we can see the distribution of the outcome "y" is not similar
# 
# ## Missing Values

# In[6]:


df_bank.isnull().sum()


# In[7]:


sns.pairplot(df_bank,hue ="y",palette="husl")


# In[8]:


dist=df_bank.hist(figsize=(12,10)) # display numerical feature distribution


# In[9]:


# Copying original dataframe
df_bank_ready = df_bank.copy()


# In[10]:


#Correlation Matrix
for i in list(df_bank.columns):
    if df_bank[i].dtype == 'object':
        df_bank[i]=pd.factorize(df_bank[i])[0]

plt.figure(figsize=(15, 5),dpi=200)
plt.title('Correlation Matrix')
sns.heatmap(df_bank.corr(),lw=1,linecolor='blue',cmap="bone_r")
plt.xticks(rotation=55)
plt.yticks(rotation = 0)
plt.show()
corr_matrix = df_bank.corr()


# ## Scale Numeric Data

# In[11]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
df_bank_ready[num_cols] = scaler.fit_transform(df_bank_ready[num_cols])

df_bank_ready.head()


# ## Encode Categorical Value

# In[12]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Encode Categorical Data
df_encoded = pd.DataFrame(encoder.fit_transform(df_bank_ready[cat_cols]))
df_encoded.columns = encoder.get_feature_names(cat_cols)

# Replace Categotical Data with Encoded Data
df_bank_ready = df_bank_ready.drop(cat_cols ,axis=1)
df_bank_ready = pd.concat([df_encoded, df_bank_ready], axis=1)

# Encode target value
df_bank_ready['y'] = df_bank_ready['y'].apply(lambda x: 1 if x == 'yes' else 0)

print('Shape of dataframe:', df_bank_ready.shape)
df_bank_ready.head()


# ## Split Dataset for Training and Testing

# In[13]:


# Select Features
feature = df_bank_ready.drop('y', axis=1)

# Select Target
target = df_bank_ready['y']

# Set Training and Testing Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, target, 
                                                    shuffle = True, 
                                                    test_size=0.15, 
                                                    random_state=1)

# Show the Training and Testing Data
print('Shape of training feature:', X_train.shape)
print('Shape of testing feature:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)


# In[14]:


from imblearn.over_sampling import RandomOverSampler
oversample=RandomOverSampler(sampling_strategy="auto")
X_over,y_over = oversample.fit_resample(X_train, y_train)


# In[15]:


y_over.value_counts()


# In[16]:


y_over.hist()


# # Modelling

# In[164]:


def evaluate_model(model, x_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(x_test)

    # Calculate accuracy, precision
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    
    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec':rec, "cm": cm,'f1': f1}

#confusion matrix
from sklearn.metrics import plot_confusion_matrix


# ## Random Forest

# In[165]:


#confusion matrix
from sklearn.metrics import plot_confusion_matrix


# In[166]:


from sklearn.ensemble import RandomForestClassifier

# Building Random Forest model 
rf = RandomForestClassifier(random_state=0)
rf.fit(X_over, y_over)


# In[167]:


# Evaluate Model
rf_eval = evaluate_model(rf, X_test, y_test)

# Print result
print('Accuracy:', rf_eval['acc'])
print('Precision:', rf_eval['prec'])
print('Recall:', rf_eval['rec'])
print('F1 Score:', rf_eval['f1'])
print('Confusion Matrix:\n', rf_eval['cm'])


# In[168]:


#Confusion Matrix for Random Forest Classifier
fig = plot_confusion_matrix(rf, X_test, y_test, display_labels=rf.classes_)
fig.figure_.suptitle("Confusion Matrix of Random Forest Classifier")
plt.grid(False)
plt.show()


# #Logistic Regression

# In[169]:


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_over, y_over)


# In[170]:


# Evaluate Model
model_eval = evaluate_model(model, X_test, y_test)

# Print result
print('Accuracy:', model_eval['acc'])
print('Precision:', model_eval['prec'])
print('Recall:', model_eval['rec'])
print('F1 Score:', model_eval['f1'])
print('Confusion Matrix:\n', model_eval['cm'])


# In[171]:


#Confusion Matrix for Logistic Regression
fig = plot_confusion_matrix(model, X_test, y_test, display_labels=model.classes_)
fig.figure_.suptitle("Confusion Matrix of Logistic Regression")
plt.grid(False)
plt.show()


# #Neural Network
# Multi-Layer Perceptron Classifier

# In[172]:


#Import MultiLayer Perceptron
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(13,10,2),
                        max_iter = 300,activation = 'relu',
                        solver = 'adam')

mlp_clf.fit(X_over,y_over)


# In[173]:


# Evaluate Model
mlp_eval = evaluate_model(mlp_clf, X_test, y_test)

# Print result
print('Accuracy:', mlp_eval['acc'])
print('Precision:', mlp_eval['prec'])
print('Recall:', mlp_eval['rec'])
print('F1 Score:', mlp_eval['f1'])
print('Confusion Matrix:\n', mlp_eval['cm'])


# In[174]:


fig = plot_confusion_matrix(mlp_clf, X_test, y_test, display_labels=mlp_clf.classes_)
fig.figure_.suptitle("Confusion Matrix of Neural Network")
plt.grid(False)
plt.show()


# In[175]:


#from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
#param_grid = {
#    'max_depth': [50, 80, 100],
#    'max_features': [2, 3, 4],
#    'min_samples_leaf': [3, 4, 5],
#    'min_samples_split': [8, 10, 12],
#    'n_estimators': [100, 300, 500]}

# Create a base model
#rf_grids = RandomForestClassifier(random_state=0)

# Initiate the grid search model
#grid_search = GridSearchCV(estimator=rf_grids, param_grid=param_grid, scoring='accuracy',
#                           cv=5, n_jobs=-2, verbose=2)

# Fit the grid search to the data
#grid_search.fit(X_train, y_train)

#grid_search.best_params_


#by running the above code we found out that the optimal parameters for random forest classifier
#are: {'max_depth': 50,
#     'max_features': 4,
#     'min_samples_leaf': 3,
#     'min_samples_split': 8,
# '   n_estimators': 500}


# In[176]:


# Building Random Forest model 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=50,max_features=4,min_samples_leaf=3,min_samples_split=8,n_estimators=500,random_state=42)
rf.fit(X_over,y_over)


# In[177]:


# Evaluate Model
rf_eval = evaluate_model(rf, X_test, y_test)

# Print result
print('Accuracy:', rf_eval['acc'])
print('Precision:', rf_eval['prec'])
print('Recall:', rf_eval['rec'])
print('F1 Score:', rf_eval['f1'])
print('Confusion Matrix:\n', rf_eval['cm'])


# In[178]:


fig = plot_confusion_matrix(rf, X_test, y_test, display_labels=rf.classes_)
fig.figure_.suptitle("Confusion Matrix of Random Forest with Optimized Parameters")
plt.grid(False)
plt.show()


# #Interpretability of Random Forest Classifier

# In[43]:


#Import shap library that is used for interpretability results
import shap


# In[59]:


#Create the explainer
explainer=shap.TreeExplainer(rf)


# In[60]:


#Show the values of X_test
X_test


# In[61]:


#We choose the first row from the test set to understand how the classifier makes predictions
X_test.loc[[3610]]


# In[124]:


pd.set_option('display.max_columns', None)
X_test.loc[[3610]]!=0.0


# In[125]:


# Calculate Shap values
choosen_instance = X_test.loc[[3610]]
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)


# In[126]:


#Let's choose some instances from the test dataset to understand to the classifier makes predictions for them.
X_test.loc[[44323]]


# In[127]:


pd.set_option('display.max_columns', None)
X_test.loc[[44323]]!=0.0


# In[128]:


# Calculate Shap values
choosen_instance = X_test.loc[[44323]]
shap_values = explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], choosen_instance)


# In[130]:


shap.summary_plot(shap_values, X_over)


# In[131]:


##Global Interpretability
sample=X_test.sample(100)
shap_values=explainer.shap_values(sample)
shap.summary_plot(shap_values,sample,max_display=10)


# In[ ]:




