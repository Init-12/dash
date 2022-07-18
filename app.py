import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

import dash

import requests
import pandas as pd
import numpy as np

import plotly.express as px

""" READ DATA """

suppress_callback_exceptions = True

response = requests.get('http://asterank.com/api/kepler?query={}&limit=2000')
df = pd.json_normalize(response.json())
df = df[df['PER'] > 0]

# CREATE STAR SIZE CATEGORY
bins = [0, 0.8, 1.2, 100]
names = ['small', 'similar', 'bigger']
df['StarSize'] = pd.cut(df['RSTAR'], bins, labels=names)

# TEMPERATURE BINS
tp_bins = [0, 200, 400, 500, 5000]
tp_labels = ['low', 'optimal', 'high', 'extreme']
df['temp'] = pd.cut(df['TPLANET'], tp_bins, labels=tp_labels)

# SIZE BINS
rp_bins = [0, 0.5, 2, 4, 100]
rp_labels = ['low', 'optimal', 'high', 'extreme']
df['gravity'] = pd.cut(df['RPLANET'], rp_bins, labels=rp_labels)

# ESTIMATE OBJECT STATUS
df['status'] = np.where((df['temp'] == 'optimal') &
                        (df['gravity'] == 'optimal'),
                        'promising', None)
df.loc[:, 'status'] = np.where((df['temp'] == 'optimal') &
                               (df['gravity'].isin(['low', 'high'])),
                               'challenging', df['status'])
df.loc[:, 'status'] = np.where((df['gravity'] == 'optimal') &
                               (df['temp'].isin(['low', 'high'])),
                               'challenging', df['status'])
df['status'] = df.status.fillna('extreme')

options = []
for k in names:
    options.append({'label': k, 'value': k})

star_size_selector = dcc.Dropdown(
    id='star-selector',
    options=options,
    value=['small', 'similar', 'bigger'],
    multi=True
)

rplanet_selector = dcc.RangeSlider(
    id='range-slider',
    min=min(df['RPLANET']),
    max=max(df['RPLANET']),
    marks={5: '5', 10: '10', 20: '20'},
    step=1,
    value=[min(df['RPLANET']), max(df['RPLANET'])]
)

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY])

""" LAYOUT """

app.layout = html.Div([

    # header
    dbc.Row(html.H1('Hello World!'),
            style={'margin-bottom': 40}),
    # filters
    dbc.Row([
        dbc.Col([
            html.Div('Select planet main semi-axis range'),
            html.Div(rplanet_selector)
        ],
            width={'size': 2}),
        dbc.Col([
            html.Div('Star Size'),
            html.Div(star_size_selector)
        ],
            width={'size': 3, 'offset': 1}),
        dbc.Col(dbc.Button('Apply', id='submit-val', n_clicks=0,
                           className='mr-2'))
    ],
        style={'margin-bottom': 40}),
    # graphics
    dbc.Row([
        dbc.Col([html.Div(id='dist-temp-chart')],
                width={'size': 6}),
        dbc.Col([html.Div(id='celestial-chart')])
    ],
        style={'margin-bottom': 40})
],
    style={'margin-left': '80px',
           'margin-right': '80px'
           }
)


""" CALLBACKS """


@app.callback(
    Output(component_id='dist-temp-chart', component_property='children'),
    Output(component_id='celestial-chart', component_property='children'),
    [Input(component_id='submit-val', component_property='n_clicks')],
    [State(component_id='range-slider', component_property='value'),
     State(component_id='star-selector', component_property='value')]
)
def update_dist_temp_chart(n, radius_range, star_size):
    chart_data = df[(df['RPLANET'] > radius_range[0]) &
                    (df['RPLANET'] < radius_range[1]) &
                    (df['StarSize'].isin(star_size))]
    if len(chart_data) == 0:
        return html.Div('Please select more data'), html.Div()

    fig1 = px.scatter(chart_data, x='TPLANET', y='A', color='StarSize')

    html1 = [html.Div('Planet Temperature ~ Distance from the Star'),
             dcc.Graph(figure=fig1)]

    fig2 = px.scatter(chart_data, x='RA', y='DEC', size='RPLANET',
                      color='status')
    html2 = [html.Div('Position on the Celestial Sphere'),
             dcc.Graph(figure=fig2)]
    return html1, html2


def update_canvas_linewidth(value):
    if isinstance(value, dict):
        return value['hex']
    else:
        return value


if __name__ == '__main__':
    app.run_server(debug=True)
