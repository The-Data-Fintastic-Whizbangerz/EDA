#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('dark')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots



# In[36]:


#Loading German Dataset
file_name = 'german_20240417.csv'

data = pd.read_csv(file_name)


# In[37]:


print("Attributes:")
for column in data.columns:
    print(column)


# In[38]:


Data_new = data.drop(data.columns[0], axis=1)
Data_new.describe()


# # Exploratory Data Analysis Of German Data

# In[39]:


def explore_object_type(df, feature_name):
    """
    Explore categorical features and returning the value counts.
    """
    if Data_new[feature_name].dtype == 'object':
        print(df[feature_name].value_counts())


# In[40]:


for feature_name in Data_new.columns:
    if Data_new[feature_name].dtype == 'object':
        print(f'\n"{feature_name}" Values with count are:')
        explore_object_type(Data_new, feature_name)


# # Data Visualations
# 

# In[9]:


import matplotlib.pyplot as plt 


# # Age Distribution

# In[57]:


numerical_attributes = ['Credit amount', 'Duration in month', 'Age', 'Installment rate in percentage of disposable income', 'disposible_income']
categorical_attributes = ['Status of existing checking account', 'Credit history', 'Purpose', 'Savings account/bonds', 'Present employment since', 'Other debtors / guarantors', 'Property', 'Other installment plans', 'Housing', 'Occupation', 'Telephone', 'Foreign worker', 'Personal status and sex', 'Sex', 'Personal status', 'occupation_new', 'number_of_existing_credits_at_this_bank', 'number_of_dependents', 'status_of_existing_checking_account', 'credit_history', 'other_installment_plans', 'housing', 'duration_group', 'disposible_income_group', 'employment_length', 'credit_amount_group']

print("Values describe: ")
print(pd.crosstab(Data_new['Age_cat'], Data_new['Outcome']))

# Defining age intervals and categories
intervals = (18, 25, 35, 60, 120)
categories = ['Student', 'Young', 'Adult', 'Senior']

# Create a new categorical column 'Age_cat' based on age intervals
Data_new['Age_cat'] = pd.cut(Data_new['Age'], intervals, labels=categories)


# Create a single subplot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot countplot of age by outcome
sns.countplot(x='Age_cat', data=Data_new, palette='hls', ax=ax, hue='Outcome',)
ax.set_title("Age Counting by Outcome", fontsize=15)
ax.set_xlabel("Age_cat")
ax.set_ylabel("Count")
ax.legend()

# Display the plot
plt.show()



# # Credit Amount Distribution_ By Group
# 

# In[55]:


print("Values describe: ")
print(pd.crosstab(Data_new['credit_amount_group'], Data_new['Outcome']))

credit_amount_group_mapping = {
    'Below_2000': 1,
    'fourthouone_6000': 2,
    'twothouone_4000': 3,
    'sixtyone_8000': 4,
    'eitthouone_10000': 5,
    'twelvethouone_14000': 6,
    'forteenthouone_16000': 7,
    'tenthouone_12000': 8,
    'Above_18001': 9
}


trace0 = go.Bar(
    x=Data_new[Data_new['Outcome'] == 1]['credit_amount_group'].value_counts().index.values,
    y=Data_new[Data_new['Outcome'] == 1]['credit_amount_group'].value_counts().values,
    name='Good Credit',
    marker=dict(
        color='LightSkyBlue',
        line=dict(width=4, color='black')
    )
)

trace1 = go.Bar(
    x=Data_new[Data_new['Outcome'] == 2]['credit_amount_group'].value_counts().index.values,
    y=Data_new[Data_new['Outcome'] == 2]['credit_amount_group'].value_counts().values,
    name='Bad Credit',
    marker=dict(
        color='DarkSlateGrey',
        line=dict(width=4, color='black')
    )
)

data = [trace0, trace1]

layout = go.Layout(
    title='Credit Amount Group Distribution by Outcome',
    xaxis=dict(
        title='Credit Amount Group',
        tickvals=list(credit_amount_group_mapping.values()),
        ticktext=list(credit_amount_group_mapping.keys())
    ),
    yaxis=dict(
        title='Count'
    ),
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)

fig.show()


# # Disposable Income Distribution by Group

# In[54]:


import plotly.graph_objects as go
print("Values describe: ")
print(pd.crosstab(Data_new['disposible_income_group'], Data_new['Outcome']))

# Define the mapping for disposable_income_group
disposable_income_group_mapping = {
    'fourthouone_6000': 1,
    'sixtyone_8000': 2,
    'eitthouone_10000': 3,
    'twelvethouone_14000': 4,
    'twothouone_4000': 5,
    'Below_2000': 6,
    'Above_18001': 7,
    'tenthouone_12000': 8,
    'sixteenthouone_18000': 9,
    'forteenthouone_16000': 10,
}

# Create bar plots for good and bad credit outcomes based on the disposable_income_group column
trace_good = go.Bar(
    x=Data_new[Data_new['Outcome'] == 1]['disposible_income_group'].value_counts().index.values,
    y=Data_new[Data_new['Outcome'] == 1]['disposible_income_group'].value_counts().values,
    name='Good Credit',
    marker=dict(
        color='LightSkyBlue',
        line=dict(width=4, color='black')
    )
)

trace_bad = go.Bar(
    x=Data_new[Data_new['Outcome'] == 2]['disposible_income_group'].value_counts().index.values,
    y=Data_new[Data_new['Outcome'] == 2]['disposible_income_group'].value_counts().values,
    name='Bad Credit',
    marker=dict(
        color='DarkSlateGrey',
        line=dict(width=4, color='black')
    )
)

# Combine the data
data = [trace_good, trace_bad]

# Define layout and add the key for the disposable_income_group mapping
layout = go.Layout(
    title='Disposable Income Group Distribution by Outcome',
    xaxis=dict(
        title='Disposable Income Group',
        tickvals=list(disposable_income_group_mapping.values()),
        ticktext=list(disposable_income_group_mapping.keys())
    ),
    yaxis=dict(
        title='Count'
    ),
    barmode='group'
)

# Create the figure
fig = go.Figure(data=data, layout=layout)

# Show the plot
fig.show()


# # Duration distribution by group

# In[53]:


import plotly.graph_objs as go
print("Values describe: ")
print(pd.crosstab(Data_new['duration_group'], Data_new['Outcome']))

# Define the mapping for the new column groups
duration_group_mapping = {
        'six_10': 1,
        'forty6_50': 2,
        'eleven_15': 3,
        'forty1_45': 4,
        'twenty1_25': 5,
        'thirty6_40': 6,
        'twenty6_30': 7,
        'fifty6_60': 8,
        'sixteen_20': 9,
        'fiftyone_55': 10,
        'thirty1_35': 11,
        'Below_5': 12,
}

# Verify that 'new_column' exists in the Data_new dataframe
if 'duration_group' in Data_new.columns:
    # Create bar plots for good and bad credit outcomes based on the new_column
    trace_good = go.Bar(
        x=Data_new[Data_new['Outcome'] == 1]['duration_group'].value_counts().index.values,
        y=Data_new[Data_new['Outcome'] == 1]['duration_group'].value_counts().values,
        name='Good Credit',
        marker=dict(
            color='LightSkyBlue',
            line=dict(width=4, color='black')
        )
    )

    trace_bad = go.Bar(
        x=Data_new[Data_new['Outcome'] == 2]['duration_group'].value_counts().index.values,
        y=Data_new[Data_new['Outcome'] == 2]['duration_group'].value_counts().values,
        name='Bad Credit',
        marker=dict(
            color='DarkSlateGrey',
            line=dict(width=4, color='black')
        )
    )

    # Combine the data
    data = [trace_good, trace_bad]

    # Define layout and add the key for the new column mapping
    layout = go.Layout(
        title='duration group Distribution by Outcome',
        xaxis=dict(
            title='duration_group Group',
            tickvals=list(duration_group_mapping.values()),
            ticktext=list(duration_group_mapping.keys())
        ),
        yaxis=dict(
            title='Count'
        ),
        barmode='group'
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Show the plot
    fig.show()
else:
    print("Column 'duration_group' does not exist in the Data_new dataframe. Please verify the column name and data.")


# # Employment Length Distribution by Group

# In[52]:


print("Values describe: ")
print(pd.crosstab(Data_new['employment_length'], Data_new['Outcome']))

employment_length_mapping = {
        'More_than_7_years': 5,
        'one_4_years': 3,
        'four_7_years': 4,
        'unemployed': 1,
        'less_than_1_year': 2,
}

if 'employment_length' in Data_new.columns:
    trace_good = go.Bar(
        x=Data_new[Data_new['Outcome'] == 1]['employment_length'].value_counts().index.values,
        y=Data_new[Data_new['Outcome'] == 1]['employment_length'].value_counts().values,
        name='Good Credit',
        marker=dict(
            color='LightSkyBlue',
            line=dict(width=4, color='black')
        )
    )

    trace_bad = go.Bar(
        x=Data_new[Data_new['Outcome'] == 2]['employment_length'].value_counts().index.values,
        y=Data_new[Data_new['Outcome'] == 2]['employment_length'].value_counts().values,
        name='Bad Credit',
        marker=dict(
            color='DarkSlateGrey',
            line=dict(width=4, color='black')
        )
    )

    data = [trace_good, trace_bad]

    layout = go.Layout(
        title='Employment length Distribution by Outcome',
        xaxis=dict(
            title='Employment length Group',
            tickvals=list(employment_length_mapping.values()),
            ticktext=list(employment_length_mapping.keys())
        ),
        yaxis=dict(
            title='Count'
        ),
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)

    fig.show()
else:
    print("Column 'employment_length' does not exist in the Data_new dataframe. Please verify the column name and data.")


# In[46]:


outcome_mapping = {
    1: 'Good',
    2: 'Bad'
}

# Cross-tabulation of Purpose and Outcome
print("Values describe: ")
print(pd.crosstab(Data_new['Purpose'], Data_new['Outcome']))

# Set figure size
plt.figure(figsize=(14, 6))

# Plot count plot for purpose count by outcome
plt.subplot(111)
g = sns.countplot(x="Purpose", data=Data_new, palette="husl", hue="Outcome")
g.set_xticklabels([purpose_mapping.get(purpose, purpose) for purpose in g.get_xticklabels()], rotation=45)
g.set_xlabel("Purpose", fontsize=12)
g.set_ylabel("Count", fontsize=12)
g.set_title("Purposes Count by Outcome", fontsize=20)
g.legend(title='Outcome', labels=[outcome_mapping[int(h)] for h in g.get_legend_handles_labels()[1]])

# Adjust spacing
plt.subplots_adjust(hspace=0.6, top=0.8)

# Show the plot
plt.show()


# # Choosing ML Model

# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[50]:


logistic_model = LogisticRegression()


# In[51]:


from sklearn.model_selection import train_test_split
#Splitting the Dataset into Train and Test
X = Data_new.drop(columns=['Outcome'])
y = Data_new['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Defining the Numerical and Categorical variables
numerical_features = ['Duration in month', 'Credit amount', 'Installment rate in percentage of disposable income',
                      'Present residence since', 'Age', 'Number of existing credits at this bank',
                      'Number of people being liable to provide maintenance for']

categorical_features = ['Status of existing checking account', 'Credit history', 'Purpose', 'Savings account/bonds',
                        'Present employment since', 'Other debtors / guarantors', 'Property', 'Other installment plans',
                        'Housing', 'Occupation', 'Telephone', 'Foreign worker', 'Personal status and sex',
                        'Personal status']

# Create a ColumnTransformer to handle preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create a pipeline with the preprocessor and logistic regression model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('log_reg', LogisticRegression(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


# In[ ]:




