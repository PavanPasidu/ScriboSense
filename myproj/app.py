import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from summarize_page import show_summarize_page
from transformers import BertTokenizer
import joblib
import torch


st.markdown("""
<style>
.css-1rs6os.edgvbvh3
{
visibility: hidden
}

.css-1lsmgbg.egzxvld0
{
visibility: hidden
}
</style>
""",unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 60px'>ScriboSense</h1>",unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center;'>The ultimate AI summary evaluater</h6>",unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Summarize","Statistics","Login"],
    icons=["vector-pen","bar-chart-fill","person-fill"],
    default_index=0,
    orientation="horizontal",
)
st.write("---")
if selected == "Summarize":
    show_summarize_page()
if selected == "Statistics":
    st.write("Statistics page")