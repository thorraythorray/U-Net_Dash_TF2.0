import os
import random

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash
import dash_core_components as dcc 
import dash_html_components as html
from dash.dependencies import Input, Output

from src.utils import mask2rle, rle2mask
from src.train import train_net
from model.config import trainCsv, trainImgPath, inferImgPath, model_path, img_size
from model.unet import unet


PAGE_SIZE = 20
basic_df = pd.read_csv(trainCsv)

app = dash.Dash(__name__)
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
app.css.config.serve_locally = False

app.layout = html.Div(children=[
    html.H1(children='Steel Defect Detection', style={'textAlign': 'center', "padding": "20px"}),

    dcc.Location(id='url', refresh=False),

    html.Div(children=[

        dcc.Link('overview', href='/overview', id="overview_id"),
        dcc.Link('mask_view', href='/mask_view', style={"marginLeft": "5%"}),
        dcc.Link('train', href='/train', style={"marginLeft": "5%"}),
        dcc.Link('infer', href='/infer', style={"marginLeft": "5%"}),
        ],
        style = {
            "display": "flex",
            "justify-content": "center"
        }
    ),

    html.Div(
        [
            dcc.Loading(
                id="loading-2",
                children=[
                    html.Div(
                        id='page-content',
                        style = {
                            "display": "flex",
                            "justify-content": "center",
                        }
                    ),
                ],
                type="cube",
            )
        ]
    ),

    # html.Div(children=[
    #     html.Button('overview', id='overview', n_clicks=0),
    #     html.Button('mask_view', id='mask_view', n_clicks=1, style={"marginLeft": "5%", "padding": "20px"}),
    #     html.Button('train', id='train', n_clicks=2, style={"marginLeft": "5%"}),
    #     html.Button('infer', id='infer', n_clicks=3, style={"marginLeft": "5%"}),
    #     ],
    #         style = {
    #         "display": "flex",
    #         "justify-content": "center"
    # }),

    # html.Div(children=[
    #         dcc.Loading(id="exec_loading", 
    #                     children=[html.Div(id="exec_loading_output")], 
    #                     type="circle",
    #                     style={
    #                         "display": "flex",
    #                         "justify-content": "center"
    #         }),
    #         html.Div(children="", id="overview_div", style={
    #             "display": "flex",
    #             "justify-content": "center"
    #         }),
    #         html.Div(children="", id="mask_view_div", style={
    #             "display": "flex",
    #             "justify-content": "center"
    #         }),
    #         html.Div(children="", id="train_div", style={
    #             "display": "flex",
    #             "justify-content": "center"
    #         }),
    #         html.Div(children="", id="infer_div", style={
    #             "display": "flex",
    #             "justify-content": "center"
    #         }),
    #     ]
    # )
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')],
              )
def display_page(pathname):
    if pathname == "/overview":
        class_dict = basic_df["ClassId"].value_counts().to_dict()
        y = [class_dict.get(i) for i in range(1, 5)]
        data = {
            'ClassId': ["class1", "class2", "class3", "class4"],
            'size': y
        }
        return html.Div(
            children=[
                html.Div(
                    'Sample Class Visualization ...',
                    style={
                        "font-size": "23px",
                        "text-align": "center",
                        "margin-top": "50px"
                    }
                ),
                dcc.Graph(id='cate_figure', figure=px.bar(data, x="ClassId", y="size",), style={"width": "100%"})
            ],
            style={
                "width": "50%"
            }
        )

    elif pathname == "/mask_view":
        r = random.randint(1, 100)
        fn = basic_df['ImageId'].iloc[r]
        img = cv2.imread(os.path.join(trainImgPath, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = rle2mask(basic_df['EncodedPixels'].iloc[r], img.shape)
        fig = px.imshow(img)
        fig_mask = px.imshow(mask)
        return html.Div(children=[
            html.Div(
                'Image Compare Bettwen Image and Mask ...',
                style={
                    "font-size": "23px",
                    "text-align": "center",
                    "margin-top": "50px"
                }
            ),
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig_mask, style={"margin-top": "-10%"})
        ],
            style={
                "width": "50%"
            }
        )

    elif pathname == "/infer":
        if not os.path.exists(model_path):
            msg = 'model.h5 not find! need to train! '
            return html.Div(msg,
                style={
                    "font-size": "23px",
                    "color": "red",
                    "text-align": "center",
                    "margin-top": "50px"
                })

        r = random.randint(1, 100)
        testfiles=os.listdir(inferImgPath)
        _infer_img = testfiles[r]

        unet_model = unet()
        unet_model.load_weights(model_path)

        infer_img = cv2.imread(os.path.join(inferImgPath, _infer_img))
        infer_img = cv2.resize(infer_img, (img_size,img_size))
        predict = unet_model.predict(np.asarray([infer_img]))

        pred_rle = []
        for img in predict:      
            img = cv2.resize(img, (1600, 256))
            tmp = np.copy(img)
            tmp[tmp<np.mean(img)] = 0
            tmp[tmp>0] = 1
            pred_rle.append(mask2rle(tmp))

        img_t = cv2.imread(os.path.join(inferImgPath, _infer_img))
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        fig = px.imshow(img_t)

        mask_t = rle2mask(pred_rle[0], img.shape)
        fig_mask = px.imshow(mask_t)

        return html.Div(children=[
            html.H3(
                'Infer Success!',
                style={
                    "font-size": "23px",
                    "text-align": "center",
                    "margin-top": "50px"
                }
            ),
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig_mask, style={"margin-top": "-10%"})
            ],
            style={
                "width": "50%"
            }
        )

    elif pathname == "/train":
        results = train_net()
        hist_df = pd.DataFrame(results.history)[['loss','val_loss']]
        # hist_df = pd.read_csv(DefectDetection_history)[['loss','val_loss']]
        fig = go.Figure(data=[
            go.Scatter(x=[1, 2], y=hist_df["loss"]),
            go.Scatter(x=[1, 2], y=hist_df["val_loss"]),
        ])

        return html.Div(children=[
            html.Div(
                    'Training Result History ...',
                    style={
                        "font-size": "23px",
                        "text-align": "center",
                        "color": "red",
                        "margin-top": "50px"
                    }
                ),
                dcc.Graph(figure=fig)
            ],
            style={
                "width": "50%"
            }
        )

    else:
        return html.Div(
            'You can test now!',
            style={
                "font-size": "23px",
                "color": "green",
                "text-align": "center",
                "margin-top": "50px"
            }
        )


#------------------------------------------------------button form--------------------------------------------------------

@app.callback(
    Output("overview_div", "children"),
    Input('overview', 'n_clicks'),
)
def overview(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'overview' in changed_id:
        class_dict = basic_df["ClassId"].value_counts().to_dict()
        y = [class_dict.get(i) for i in range(1, 5)]
        data = {
            'ClassId': ["class1", "class2", "class3", "class4"],
            'size': y
        }
        return dcc.Graph(id='cate_figure', figure=px.bar(data, x="ClassId", y="size",), style={"width": "50%"})


@app.callback(
    Output("mask_view_div", "children"),
    Input('mask_view', 'n_clicks')
    )
def mask_view(btn2):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'mask_view' in changed_id:
        
        r = random.randint(1, 100)
        fn = basic_df['ImageId'].iloc[r]
        img = cv2.imread(os.path.join(trainImgPath, fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = rle2mask(basic_df['EncodedPixels'].iloc[r], img.shape)
        fig = px.imshow(img)
        fig_mask = px.imshow(mask)
        return html.Div(children=[
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig_mask, style={"margin-top": "-10%"})
        ])


@app.callback(
    Output("train_div", "children"),
    Input('train', 'n_clicks'),
    )
def train(btn3):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'train' in changed_id:
        results = train_net()
        hist_df = pd.DataFrame(results.history)[['loss','val_loss']]
        # hist_df = pd.read_csv(DefectDetection_history)[['loss','val_loss']]
        fig = go.Figure(data=[
            go.Scatter(x=[1, 2], y=hist_df["loss"]),
            go.Scatter(x=[1, 2], y=hist_df["val_loss"]),
        ])

        return html.Div(children=[
            dcc.Graph(figure=fig)
        ])


@app.callback(
    Output("infer_div", "children"),
    Input('infer', 'n_clicks')
    )
def infer(btn4):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'infer' in changed_id:
        print(model_path)
        if not os.path.exists(model_path):
            msg = 'model.h5 not find! Perhaps model has not trained! '
            return html.Div(msg)

        r = random.randint(1, 100)
        testfiles=os.listdir(inferImgPath)
        _infer_img = testfiles[r]

        unet_model = unet()
        unet_model.load_weights(model_path)

        infer_img = cv2.imread(os.path.join(inferImgPath, _infer_img))
        infer_img = cv2.resize(infer_img, (img_size,img_size))
        predict = unet_model.predict(np.asarray([infer_img]))
        
        pred_rle = []
        for img in predict:      
            img = cv2.resize(img, (1600, 256))
            tmp = np.copy(img)
            tmp[tmp<np.mean(img)] = 0
            tmp[tmp>0] = 1
            pred_rle.append(mask2rle(tmp))

        img_t = cv2.imread(os.path.join(inferImgPath, _infer_img))
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        fig = px.imshow(img_t)

        mask_t = rle2mask(pred_rle[0], img.shape)
        fig_mask = px.imshow(mask_t)

        return html.Div(children=[
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig_mask, style={"margin-top": "-10%"})
        ])


if __name__ =="__main__":
    app.run_server()
