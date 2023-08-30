# import pandas as pd
# import numpy as np
from transformers import BertTokenizer
import joblib
import torch
import streamlit as st

def tokenize_content(question,title,text,summary):
    # Tokenizer for the bert model
    tokenizer = BertTokenizer.from_pretrained('Models/Tokenizer')
    tokenized_inputs = tokenizer(
    question=question,
    title=title,
    text=text,
    summary=summary,
    padding=True,  # Specify padding here
    truncation=True,
    max_length=16,
    return_tensors='pt'
    )

    input_ids = tokenized_inputs['input_ids']
    attention_mask = tokenized_inputs['attention_mask']

    return input_ids,attention_mask

def predict_content(question,title,text,summary):

    # predict function for content
    model = joblib.load('Models/bertforcontent.pkl') # load the content model
    model.eval()

    input_id,attention_mask=tokenize_content(question,title,text,summary)

    with torch.no_grad():
        outputs = model(input_id,attention_mask=attention_mask)
        predicted_score = outputs.logits.item()

    return predicted_score



# --------------------------------------------------------------------



