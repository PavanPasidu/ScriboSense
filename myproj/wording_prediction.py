import torch
import numpy as np
import joblib
import streamlit as st

# # with open('wording_model.pkl', 'rb') as file:                  
# #     data = pickle.load(file)

# data = joblib.load("wording_model/wording_model_1.pkl")

# regressor_loaded = data.regressionModel
# model=data.gpt2_model
# tokenizer=data.tokenizer

# def score_sentence(sentence, model, tokenizer):
#     tokenize_input = tokenizer.encode(sentence)
#     tensor_input = torch.tensor([tokenize_input])
#     loss = model(tensor_input, labels=tensor_input)[0]
#     return np.exp(loss.detach().numpy())


# # Get a sentence from user input 
# user_input = 'the diffrent social class were like the diffrent part of the pyramid aka the govern if you were in the high class you are at the top of the pyramid lower class your the bottom of the pyramid or the base' #replace with input(summary) from app

# # Get the score for the input sentence
# sentence_score = score_sentence(user_input, model, tokenizer)

# X = np.array([[sentence_score]])

# wording  = regressor_loaded.predict(X)

data = joblib.load("wording_model/wording_model_1.pkl")

regressor_loaded = data.get("regressionModel")
model_2=data.get("gpt2_model")
tokenizer=data.get("tokenizer")

def score_sentence(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss=model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())


# Get a sentence from user input 
user_input = "this project is annoyingthis project is annoying this project is annoying this project is annoying  "

# Get the score for the input sentence
sentence_score = score_sentence(user_input, model_2, tokenizer)

X = np.array([[sentence_score]])

wording  = regressor_loaded.predict(X)



#---------------------------------------------------------------------------------------------------

st.write("Hi")
st.write(wording)
