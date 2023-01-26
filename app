"""
Module7 Project
Solar panel price monitoring app
"""

import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor



st.set_page_config(
    page_title="Solar Panel Prices", page_icon="🖼️", initial_sidebar_state="collapsed"
)
st.markdown("# Solar Panel Prices")


# ---------------------------------------------------------------------------
# PART 1 Solar Panel Dataset ----------------------------------------------
# ---------------------------------------------------------------------------

st.image("solar_panel.png")

st.header("Overview")

st.write(
'''
**List of all solar panels**

''')


#data = pickle.load(open('solar_clean.pickle', 'rb'))
data = pd.read_csv("solar_clean_new.csv", index_col=0)
#data = pd.read_pickle("solar_clean.pickle")

data = data.rename(columns={'brand_bucketed': 'brand',
                            'panel_type_bucketed': 'panel_type',
                            'region_bucketed': 'region',
                            })

data.drop(['power_wp', 'dimension_sqm'], axis=1, inplace=True)

# drop NA's again just in case
data_clean = data.dropna()


if st.checkbox('Show dataset'):
    data

    
# ---------------------------------------------------------------------------
# PART 2 Visualizations
# ---------------------------------------------------------------------------

    
st.write(
'''
**Manufacturing regions**
'''
)


# histogram of regions    
fig, ax = plt.subplots(figsize=(10,10))
sns.histplot(data=data_clean, y="region", color='orange')
ax.set_ylabel('Region')

if st.checkbox('Show regions'):
    st.pyplot(fig)

# show min, max, and average price
# https://towardsdatascience.com/streamlit-from-scratch-presenting-data-d5b0c77f9622

st.write(
'''
**Price range**
'''
)

minprice = min(data_clean['price_euro'])
maxprice = max(data_clean['price_euro'])
meanprice = np.round(np.mean(data_clean['price_euro']), 2)

col1, col2, col3 = st.columns([33,33,33])

col1.metric("Cheapest panel", minprice,
            help="Price in Euro")
col2.metric("Most expensive panel", maxprice, 
            help="Price in Euro"
            )
col3.metric("Average price", meanprice, 
            help="Price in Euro"
            )


# ---------------------------------------------------------------------------
# PART 3 - Model and Predictions from Input
# ---------------------------------------------------------------------------

st.header("Price Predictions")

# get features and target (only if training takes place in app)
X = data_clean.drop(['price_euro'], axis=1)
y = data_clean['price_euro']

X_cat = list(X.drop(['efficiency_percent', 'weight_kg'], axis=1))

# preprocessing pipeline (only if training in app)
pipeline = ColumnTransformer([("cat", OneHotEncoder(drop='first'), X_cat)],
                             remainder='passthrough'
                             )
    
X_prepped = pipeline.fit_transform(X)


# load pre-trained model
#pretrained_rfr = pickle.load(open('rfr_model', 'rb'))
pretrained_rfr = pd.read_pickle("rfr_model")


# train model (only if training here in app)
#cache this function 
#@st.cache
#def fit_model(model, X, y):
#    model = model
#    model_fitted = model.fit(X,y)
#    return model_fitted

#rfr_fitted = fit_model(rfr, X_train, y_train)
#rfr_fitted = rfr.fit(X_prepped, y)    


st.write(
    '''
    **Choose your specifications:**
    '''
    )

eff = st.number_input('Efficiency (in %)', value=21)
brand = st.selectbox('Brand', X['brand'].unique(), index=0)
panel = st.selectbox('Panel', X['panel_type'].unique(), index=0)
weight = st.number_input('Weight (in kg)', value=18)
region = st.selectbox('Region', X['region'].unique(), index=0)

input_data = pd.DataFrame({'efficiency_percent': [eff],
                           'brand': [brand],
                           'panel_type': [panel],
                           'weight_kg': [weight],
                           'region': [region]
                           })

x = pipeline.transform(input_data)

pred = pretrained_rfr.predict(x)[0]



st.write(
'''
 ### Expected price:
'''
f'EUR  {np.round(float(pred), 3):,}'
)


