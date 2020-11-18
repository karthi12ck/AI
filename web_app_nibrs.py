# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:07:15 2020

@author: karthickk
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from autocorrect import Speller
data=pd.read_csv(r'E:\nibrs\backup\back_up_full\nibrs.csv')
import gensim
from gensim import corpora
from pprint import pprint
from gensim.utils import simple_preprocess
import re
model=keras.models.load_model('nibrs_class.h5')
dictionary=corpora.Dictionary.load('nibrs_class.dict')
t=Speller()
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'wii", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"st","street",text)
    text = re.sub(r"domestreetic","domestic",text)
    text = re.sub(r'br','',text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text
data1=dictionary.token2id
inv_map = {v: k for k, v in data1.items()}
import pandasql as ps
cfs_output_lookup=pd.read_csv('cfs_output_lookup.csv')

#user_input = ['LT REVOLINSKY REQUESTED A DEPUTY TO 11-98 IN THE AREA OF C/R 17 WEST OF C/R H FOR A POSSIBLE VANDALISM REPORT. DEPUTY GIBBS ON SCENE WITH L-1. L-1 EXPLAINED THAT 3 MALE JUVENILES THREW 2 BOTTLES IN THE ROADWAY, IN FRONT OF A RESIDENCE DRIVEWAY. THE OWNER/RP, JA']
'''user_input=data.Txt.iloc[20]
user_input=[user_input]

clean_doc = []
for doc in user_input:
    clean_doc.append(clean_text(doc))
clean_doc =np.array(clean_doc)
spell_check=[]
for doc in clean_doc:
    
    spell_check.append(t(doc))
spell_check=np.array(spell_check)
clean_doc=spell_check
tokenized_doc= [simple_preprocess(doc) for doc in clean_doc]
tokenized_doc=np.array(tokenized_doc)
encoded_doc1 = []

for doc in tokenized_doc:
    encoded_doc1.append(dictionary.doc2idx(doc))
encoded_doc1=np.array(encoded_doc1) 
result=[]
for i in encoded_doc1[0]:
    c=inv_map.get(i)
    
    result.append(c)
#c=encoded_doc1     

temp=list(encoded_doc1[0])
for index ,value in enumerate(temp):
    if value == -1:
        temp[index] = 0
temp=np.array(temp)
temp=np.expand_dims(temp,0)
encoded_doc1=temp
AI_encoded_input = sequence.pad_sequences(encoded_doc1, maxlen=100,value= 0)
print(user_input)
print('\n')
print(result)
print('\n')
print(encoded_doc1)












result=model.predict_classes(AI_encoded_input)
tem=cfs_output_lookup[cfs_output_lookup.x_code == result[0]]
predicted_CFS_class=tem.cnvDESC.values[0]
print('The predicted CFS_class for given data is {}'.format(predicted_CFS_class))'''



app = dash.Dash()

app.layout = html.Div(
        
   [html.Div([
    html.H1("AI CFS CLASS PREDICTION",style={
            'textAlign': 'center'
            
        }),html.Label('ENTER ROW NUMBER OF DATA', style={"font-weight": "bold"}),
    dcc.Input(
        id='row',
        placeholder='ENTER THE ROW NUMBER',
        type="number",
        value=None,
        
    ),html.Button('Submit', id='submit-val',n_clicks=0),
      html.Br(),
    html.Br(),
    html.Label('INPUT TEXT', style={"font-weight": "bold"}),
    html.Div(id='inputtext',style={'padding':10}),
    html.Br(),
    html.Br(),
    html.Label('ACTUAL CFS CLASS', style={"font-weight": "bold"}),
    html.Div(id='actual',style={'padding':10}),
    html.Br(),
    html.Br(),
    html.Label('PREDICTED CFS CLASS', style={"font-weight": "bold"}),
    html.Div(id='output',style={'padding':10})])])
    
@app.callback(
    Output('inputtext', 'children'),
    
    
     [dash.dependencies.Input('submit-val','n_clicks')],
     state=[dash.dependencies.State('row','value')]
    
   
)

def update_inputtext(n_clicks,value):
    res=data.Txt.iloc[value]
    return res


@app.callback(
    Output('actual', 'children'),
    
    
     [dash.dependencies.Input('submit-val','n_clicks')],
     state=[dash.dependencies.State('row','value')]
    
   
)

def update_inputtext(n_clicks,value):
    res=data.cnvDESC.iloc[value]
    return res


@app.callback(
    Output('output', 'children'),
    
    
     [dash.dependencies.Input('submit-val','n_clicks')],
     state=[dash.dependencies.State('row','value')]
    
   
)

def update_inputtext(n_clicks,value):
    #data1=dictionary.token2id
    #inv_map = {v: k for k, v in data1.items()}
    user_input=data.Txt.iloc[value]
    user_input=[user_input]
    clean_doc = []
    for doc in user_input:
        clean_doc.append(clean_text(doc))
    clean_doc =np.array(clean_doc)
    spell_check=[]
    for doc in clean_doc:
    
        spell_check.append(t(doc))
    spell_check=np.array(spell_check)
    clean_doc=spell_check
    tokenized_doc= [simple_preprocess(doc) for doc in clean_doc]
    tokenized_doc=np.array(tokenized_doc)
    encoded_doc1 = []

    for doc in tokenized_doc:
        encoded_doc1.append(dictionary.doc2idx(doc))
    encoded_doc1=np.array(encoded_doc1) 
    #result=[]
    #for i in encoded_doc1[0]:
        #c=inv_map.get(i)
    
    #result.append(c)
#c=encoded_doc1     

    temp=list(encoded_doc1[0])
    for index ,value in enumerate(temp):
        if value == -1:
            temp[index] = 0
    temp=np.array(temp)
    temp=np.expand_dims(temp,0)
    encoded_doc1=temp
    AI_encoded_input = sequence.pad_sequences(encoded_doc1, maxlen=100,value= 0)
#print(user_input)
#print('\n')
#print('\n')
#print(encoded_doc1)
    result=model.predict_classes(AI_encoded_input)
    tem=cfs_output_lookup[cfs_output_lookup.x_code == result[0]]
    predicted_CFS_class=tem.cnvDESC.values[0]
    return predicted_CFS_class































    
 
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False)   
    