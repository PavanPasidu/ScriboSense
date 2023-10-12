import streamlit as st
from transformers import BertTokenizer
import joblib
import torch
from predict_page import tokenize_content
from predict_page import predict_content


def display_result(content_score,wording_score):
    content_score = round(content_score,3)
    wording_score = round(wording_score,3)
    col1,col2=st.columns(2)
    col1.metric("Content",content_score,delta=None)
    col2.metric("Wording",wording_score,delta=None)

def show_summarize_page():
    
    title = st.text_input("Enter the Title")
    prompt = st.text_input("Enter the Prompt")
    text = st.text_area("Enter the Text")
    summary = st.text_area("Enter your summary")
    clicked = st.button("submit")
    st.write("---")
    if clicked:
        # input title,prompt,text,summary to the model
        # get predicted values
        content_score = predict_content(question=prompt,title=title,text=text,summary=summary)
        display_result(content_score,5) # replace the values with content and wording scores predicted by ML model


#----------------------------------------------------------------------------------------------------------------



