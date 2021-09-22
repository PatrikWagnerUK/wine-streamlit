import pandas as pd
import numpy as np
import streamlit as st
import pickle
import boto3
from s3fs.core import S3FileSystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, sigmoid_kernel

## Importing and cleaning data

#predictors = pd.read_csv('wine_pred_matrix.csv')

## Loading model from pickle file
#@st.cache(allow_output_mutation=True)
#def load_model():
#    with open('wine_model.pkl', 'rb') as file:
#        data = pickle.load(file)
#    return data

s3 = boto3.resource('s3', aws_access_key_id=st.secrets["key_id"], aws_secret_access_key=st.secrets["secret_key"])
csv_url = 'https://wineproj.s3.us-east-2.amazonaws.com/wine_pred_matrix_ita.csv'
data = pickle.loads(s3.Bucket("wineproj").Object("wine_model_ita.pkl").get()['Body'].read())
predictors = pd.read_csv(csv_url)

#data = load_model()
sig_kern = data["model"]

## Creating function to display streamlit page
def show_page():
    st.title('Wine Recommendation Generator 1.0')
    st.subheader('Finding you the highest ranking and most similar wines backed by 6,000+ sommelier reviews.')
    st.markdown("____")

def recommend_wine(sig_kern=sig_kern):
    variety = st.sidebar.selectbox("Filter Wines by variety:", np.unique(predictors['variety']))
    variety_filtered = predictors[(predictors['variety'] == variety)]
    region = st.sidebar.selectbox("Filter Wines by region:", np.unique(variety_filtered['region_1']))
    region_filtered = variety_filtered[(variety_filtered['region_1'] == region)]
    #st.dataframe(variety_filtered[['name', 'variety']])
    user_wine_input = st.selectbox('Recommend me a wine similar to the:', variety_filtered['name'].sort_values(ascending=True))


    index = pd.Series(predictors.index, index=predictors['name']).drop_duplicates()
    indx = index[user_wine_input]
    sigmoid_score = list(enumerate(sig_kern[indx]))
    sigmoid_score = sorted(sigmoid_score, key = lambda x:x[1], reverse = True)
    sigmoid_score = sigmoid_score[1:4]
    position = [i[0] for i in sigmoid_score]

    pd.set_option('display.max_colwidth', None)
    if st.button("Recommend Wine"):
        st.header(f"Other wines to consider are: ")

        name1 = predictors[['name']].iloc[position[0]]
        desc1 = predictors[['description']].iloc[position[0]]
        s_url1 = predictors[['search_url']].iloc[position[0]].to_string(header=False, index=False), unsafe_allow_html=True)
        st.subheader("The " + name1.to_string(header=False, index=False))
        st.markdown(desc1.to_string(header=False, index=False))
        st.markdown('[Purchase Wine]' + s_url1)
        st.markdown("____")

        name2 = predictors[['name']].iloc[position[1]]
        desc2 = predictors[['description']].iloc[position[1]]
        s_url2 = predictors[['search_url']].iloc[position[1]].to_string(header=False, index=False), unsafe_allow_html=True)
        st.subheader("The " + name2.to_string(header=False, index=False))
        st.markdown(desc2.to_string(header=False, index=False))
        st.markdown('[Purchase Wine]' + s_url2)
        st.markdown("____")

        name3 = predictors[['name']].iloc[position[2]]
        desc3 = predictors[['description']].iloc[position[2]]
        s_url3 = predictors[['search_url']].iloc[position[2]].to_string(header=False, index=False), unsafe_allow_html=True)
        st.subheader("The " + name3.to_string(header=False, index=False))
        st.markdown(desc3.to_string(header=False, index=False))
        st.markdown('[Purchase Wine]' + s_url3)
