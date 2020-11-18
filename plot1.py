# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:24:11 2020

@author: KarthickK
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import pandasql as ps


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([html.H1(children='USE CASE 2-Provide Filter based on CFSClass wise prediction',style={
            'textAlign': 'center',
           
        },
),html.H2('input = cfs_class                    output = location')
,    html.Br(),
    html.Button('Submit', id='submit-val',n_clicks=0),
    dcc.Graph(
        id = 'indicator-graph',
        figure={
        'layout': {'title':'cfs class wise prediction'}
        }),
    html.H2('input = location                  output = cfs_class'),
    dcc.Graph(
        id = 'indicator-graph1',
        figure={
        'layout': {'title':'location prediction'}
        }),html.H1(children='USE CASE 4-actual vs predicted output'),
       dcc.Graph(
        id = 'indicator-graph2',
        figure={
        'layout': {'title':'location prediction'}
        }) ,
        html.H1(children='USE CASE 5 -Criminal detector model output'),
       dcc.Graph(
        id = 'indicator-graph3',
        figure={
        'layout': {'title':'location prediction'}
        }),html.H1(children='USE CASE 5 -Pattern graph for a person associated with incident'),
       dcc.Graph(
        id = 'indicator-graph4',
        figure={
        'layout': {'title':'location prediction'}
        })
])





@app.callback(
    Output('indicator-graph', 'figure'),
    [dash.dependencies.Input('submit-val','n_clicks')])

def update_graph(n_clicks):
    import plotly.express as px
    user_final_input=pd.read_csv('userfinal.csv')
    px.set_mapbox_access_token("pk.eyJ1Ijoia2FydGhpMTJjayIsImEiOiJjazNmbDRtZjUwMDBzM2lvNmN3N2tuZndtIn0.3iW4aEQ6koKKe8cpOja0hg")
    fig = px.scatter_mapbox(user_final_input, lat="latitude", lon="longitude",  hover_data=["CADCFSDesc",'streetname','Location'],color='CADCFSDesc',

                         zoom=11, height=800)
    fig.update_layout(mapbox_style="open-street-map",mapbox_center_lon=-88,mapbox_center_lat=43.1)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@app.callback(
    Output('indicator-graph1', 'figure'),
    [dash.dependencies.Input('submit-val','n_clicks')])

def update_graph(n_clicks):
    import plotly.express as px
    final=pd.read_csv('final.csv')
    fig = px.scatter_mapbox(final, lat="latitude", lon="longitude",  hover_data=["original_CADCFSDesc", "predicted_CADCFSDesc",'accuracy_of_prediction'],
                        color="predicted_CADCFSDesc",zoom=3,animation_group='predicted_crime',height=800)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig
 
    

@app.callback(
    Output('indicator-graph2', 'figure'),
    [dash.dependencies.Input('submit-val','n_clicks')])


def update_graph(n_clicks):
    import plotly.express as px
    final=pd.read_csv('final.csv')
    fig = px.scatter_mapbox(final, lat="latitude", lon="longitude",  hover_data=["original_CADCFSDesc", "predicted_CADCFSDesc",'accuracy_of_prediction'],
                        color="accuracy_of_prediction",zoom=3,animation_group='predicted_crime',height=800)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

@app.callback(
    Output('indicator-graph3', 'figure'),
    [dash.dependencies.Input('submit-val','n_clicks')])


def update_graph(n_clicks):
    import plotly.express as px
    criminal_to_look_out_2017=pd.read_csv('criminal.csv')
    import plotly.express as px

    fig = px.scatter_mapbox(criminal_to_look_out_2017, lat="latitude", lon="longitude",  hover_data=["CADCFSDesc", "FullName",'Height','Weight','Age','crime_count'],color='CADCFSDesc',

                         zoom=11, height=800)
    fig.update_layout(mapbox_style="open-street-map",mapbox_center_lon=-88,mapbox_center_lat=43.1)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig

@app.callback(
    Output('indicator-graph4', 'figure'),
    [dash.dependencies.Input('submit-val','n_clicks')])


def update_graph(n_clicks):
    import plotly.express as px
    criminal_to_look_out_2017=pd.read_csv('criminal.csv')
    criminal_to_look_out_2017.drop(columns='race',axis=0,inplace=True)
    #import plotly.express as px
    #criminal_to_look_out_2017=ps.sqldf('select  top 100 * from criminal_to_look_out_2017')
    criminal_to_look_out_2017=criminal_to_look_out_2017.head(25)
    fig = px.parallel_categories(criminal_to_look_out_2017, dimensions=['FullName', 'maritalstatusdesc','Height', 'Weight','RaceDesc','CADCFSDesc'],
                color="cadcfsclass_prediction", color_continuous_scale=px.colors.sequential.Inferno,
               )
    return fig
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False)   