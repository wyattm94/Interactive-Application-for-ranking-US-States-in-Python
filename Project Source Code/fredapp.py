'''
______________________________________________

Script  : fredapp.py
Root    : finapp/root_app
Author  : Wyatt Marciniak
______________________________________________
'''

# region :: Dependancies -----

import sys
import os
import math
import xlrd

from helper import *
from fred_api import *
from app_utilitiles import *
# from custom_classes import *
from StatsCore import *
import numpy as np
import pandas as pd
# import openpyxl as opxl
import dash
import dash_daq as daq
# dash.__version__
import dash_renderer
import dash_table as dtbl
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_core_components as dcc
import dash_html_components as h
from dash.dependencies import Output as ddo
from dash.dependencies import Input as ddi
from dash.dependencies import State as dds
import datetime as dt
from datetime import datetime as dtt
#import plotly
#import plotly.plotly as ppy
import chart_studio
import chart_studio.plotly as ppy
#from plotly import graph_objs as go
from chart_studio import graph_objs as go
import multiprocessing.pool as mpool
from multiprocessing.pool import ThreadPool as mtpool
from multiprocessing import Process, Queue
from itertools import product
import openpyxl as opx
# import flask
# from app import app as f_app
# from proxy_mask import *
# endregion -----

# (1) Calculate average dates (over time periods) --> Get rank and score
# (2) Determine selected and non-selected states in table

def cscore(d,rid,byrank=False,mag=1000000):
    if byrank: temp = d['rank']
    else: temp = d['score']
    # Parse Dates
    rid = [x for x in dates if int(x) in rid]
    temp.index = temp['dateu']
    sub = temp.loc[rid]
    print('cscore index (dates) values:\n')
    print(list(rid)[0],list(rid)[-1])
    print(list(sub.index)[0],list(sub.index)[-1])
    return [(cSum(list(sub[x])) / len(sub.index))/mag for x in states]

def helper_ss(u,c,a,s):
    if not all([isok(x) for x in [u, c, a, s]]) or max(u, c, a) == a:
        ss = [x for x in range(0, 50)];ns = []
    elif max(u,c,a) == c:
        ss = []; ns = [x for x in range(0,50)]
    else: ss = [x for x in s]; ns = [x for x in range(0, 50) if x not in ss]

    return dict(
        cellcond = unlist([
            [{'if': {'row_index': s}, 'background-color': '#9effae'} for s in ss],
            [{'if': {'row_index': s}, 'background-color': '#ffd1d1'} for s in ns]]),
            # [{'if': {'row_index': -1}, 'background-color': 'blue'}]]),
        ss = ss,
        ns = ns
    )

def helper_dref2tag():
    dtag = []
    for x in list(dref['tag']):
        y = unlist(x.split(' '));
        y = y[:-2]
        x = ''
        for i in y: x += mkstr(' ', i, ' ')
        dtag.append(x)
    return dtag

# cSum([1,2,3])
# mkstr([1,2,3],sep_list='+')

# region >> Initialize
# Initialize App - block callback (non-essential)
app = dash.Dash('__main__',assets_folder='fred_assets')
app.config.supress_callback_exceptions = True
app.config.update({ 'routes_pathname_prefix':'','requests_pathname_prefix':''})


itime0 = time.time()
dset,dref,us_dict,states,weights = fredloader()
dref.index = dref['id']; dref = dref.loc[list(dset.keys())]
states = [x.lower() for x in states]
dates = [x for x in dset[list(dset.keys())[0]]['dateu']]
weights = weights.iloc[:-4,1:]
score_data0 = calcScoreData(dset,unlist([['date','dateu'],states]),weights)

# Starting Dates
t0 = dtt(2008,1,1)
t1 = dtt(2017,1,1)

dw_id = sorted([int(list(weights['dataset'])[x].split('_')[1]) for x in range(len(weights.index))])


# Initial Drop Down Menus
dd_state_sort = create_options(
    [
        'States (Asc)','States (Desc)',
        'Ranks (Asc)','Ranks (Desc)',
        'Scores (Asc)','Scores (Desc)'
    ],
    [
        'State_True','State_False',
        'Rank_True','Rank_False',
        'Score_True','Score_False'
    ])
dd_param_sort = create_options(
    [
        'Names (Asc)','Names (Desc)',
        'Weights (Asc)','Weights (Desc)',
        'ID (Asc)','ID (Desc)'
    ],
    [
        'Parameter_True','Parameter_False',
        'Weight_True','Weight_False',
        'ID_True','ID_False'
    ]
)
dd_param_weights = create_options(dw_id,dw_id)

data_tag_clean = []
for x in list(dref['tag']):
    y = unlist(x.split(' ')); y = y[:-2]; x = ''
    for i in y: x += mkstr(' ',i)
    data_tag_clean.append(x)

# dd_param_f0 = create_options(
#     data_tag_clean,
#     [int(x.split('_')[1]) for x in list(dref['id'])]
# )
# dd_param_f1 = create_options(
#     ['(is)','(not)'],
#     ['','not']
# )
# dd_param_f2 = create_options(
#     [ '==','>','>=','<','<=','in'],
#     [ '==','>','>=','<','<=','in']
# )


sysout('\n[FULL INITIALIZATION TIME (in seconds): ',round(time.time()-itime0,3),'\n\n ')
del itime0

# endregion

### APP LAYOUT
app.layout = h.Div(className='appcontent',children=[

    # Header
    h.Header(className='header', id='app_header',
             children=[
                 h.H1(className='', id='hh1',
                      children=[
                          'Welcome to "Comparing States to live in (Economically)"'
                      ]),
                 h.P(className='', id='hpara',
                     children=[
                         "Alter the state/parameter subsets/sorting, time range and weight modifiers (Scoring) - "
                         "Current Version : v2.0.0 ('Eddy') - Stable"
                     ]),
                    # Hidden Div elems
                    h.Div(className='hidden_div', id='hidden_currp', style={'display': 'none'}),
                    h.Div(className='hidden_div', id='hidden_score', style={'display': 'none'})
    ]),

    # region >> Main Row Parent
    h.Div(className='row',id='mainp',
          children=[

              # region >> (L) Param Controls
              h.Div(className='row',id='mainleftP',
                    children=[

                        # region >> (L - Top) Date Range Selection (Above Param Tables)
                        h.Div(className='row',id='daterangep',
                              children=[
                                    h.H1(className='dph', id='h_daterange',
                                        children=['Date Range: ']),
                                    dcc.DatePickerRange(
                                        className='abovetables', id='daterange',
                                        display_format='MM YYYY',
                                        start_date=t0, end_date=t1,
                                        min_date_allowed=t0, max_date_allowed=t1,
                                        start_date_placeholder_text='T_0',
                                        end_date_placeholder_text='T_T', ),

                                    h.Button(className='btn', id='pbtn_rescore',
                                             children=['RESCORE'], n_clicks=0,
                                             n_clicks_timestamp=0),
                                    h.Button(className='btn', id='pbtn_restore',
                                             children=['RESTORE'], n_clicks=0,
                                             n_clicks_timestamp=0)
                        ]),
                        # endregion << L-top Date Range

                        # region >> (L - Mid) Tables >> States and Params (KEY ELEM)
                        h.Div(className='row',id='sptablep',
                              children=[

                                  # region >> (L - Mid - Table 1 - States)
                                  h.Div(className='mainleft', id='colstate',
                                        children=[
                                            h.H1(className='tbl_header',id='h_state_table',
                                                 children=['States, Ranks and Scores']),
                                            h.Div(className='tbl_holder', id='tblp_state',
                                                  children=[dtbl.DataTable(id='state_table')]),
                                            h.Div(className='sbtn', id='sbtn_p',
                                                  children=[
                                                      h.Button(className='btn', id='sbtn_addall',
                                                               n_clicks=0, n_clicks_timestamp=0, children=['All']),
                                                      h.Button(className='btn', id='sbtn_rmvall',
                                                               n_clicks=0, n_clicks_timestamp=0, children=['Clear']),
                                                  ]),
                                            h.Div(className='sbtn',id='sdd_p',
                                                  children=[
                                                      dcc.Dropdown(className='dd', id='ddstatesort',
                                                                   clearable=False,
                                                                   # placeholder='Sort States',
                                                                   options=dd_state_sort,
                                                                   value=dd_state_sort[0]['value'])
                                            ])



                                  ]),
                                  # endregion - left mid table 1 (states)

                                  # region >> (L - Mid - Table 2 - Parameters)
                                  h.Div(className='mainleft', id='colparam',
                                        children=[
                                            h.H1(className='tbl_header', id='h_param_table',
                                                 children=['Parameters and Weights']),
                                            h.Div(className='tbl_holder', id='tblp_param',
                                                  children=[dtbl.DataTable(id='param_table')]),
                                            h.Div(className='pbtn', id='pbtn_p',
                                                  children=[
                                                        h.Button(className='btn', id='pbtn_addall',children=['All'],
                                                                 n_clicks=0,n_clicks_timestamp=0),
                                                        h.Button(className='btn', id='pbtn_rmvall',children=['Clear'],
                                                                 n_clicks=0,n_clicks_timestamp=0)
                                                  ]),
                                            h.Div(className='pbtn', id='pddsort_p',
                                                  children=[
                                                        dcc.Dropdown(className='dd', id='ddparamsort',
                                                                     clearable=False,
                                                                     # style={'height':'100px'},
                                                                     # placeholder='Sort Parameters',
                                                                     options=dd_param_sort,
                                                                     value=dd_param_sort[0]['value'])
                                            ]),
                                            h.Div(className='ddw',id='ddwp',
                                                  children=
                                                  [
                                                      h.Div(className='ddw_sel', id='ddw_selp',
                                                            children=[
                                                                    dcc.Dropdown(className='dds', id='ddparamweight',
                                                                                 clearable=False,
                                                                                 # placeholder='Sort Parameters',
                                                                                 options=dd_param_weights,
                                                                                 value=dd_param_weights[0]['value']),
                                                            ]),
                                                      h.Div(className='ddw_set',id='ddw_setp',
                                                            children=[
                                                                    dcc.Input(className='ddw',id='ddw_in',
                                                                              # placeholder='--'
                                                                                type='Input',
                                                                                value=dd_param_weights[0]['value'])
                                                            ]),
                                                      h.Div(className='ddw_btn',id='ddw_btnp',
                                                            children=[
                                                                    h.Button(className='ddw', id='ddw_chg',children=['OK'],
                                                                             n_clicks=0,n_clicks_timestamp=0),
                                                                    h.Button(className='ddw', id='ddw_clr', children=['Clear'],
                                                                             n_clicks=0, n_clicks_timestamp=0)
                                                            ])
                                                  ])
                                  ])
                                  # endregion - left mid table 2 (params)
                        ]),
                        # endregion - left mid tables

                        # region >> (L - Bottom) Weight/Filters/Analysis Controls
                        h.Div(className='row',id='wfa_p',
                              children=[

                        ])
                        # endregion << (L)
              ]),
              # endregion >> param control

              # region >> (R) Map
              h.Div(className='row', id='mainmidP',
                    children=[
                        h.Div([dcc.Graph(id='state_map')], className='mapp_states'),
              ]),
              # endregion << Map
    ]),
    # endregion - main row

    h.Footer(className='footer', id='app_footer',
             children=[
                 h.H1(className='hfooter',id='footer_h',children=['Thank your using the Application']),
                 h.P(className='pfooter',id='footer_p',children=['--'])
             ])

])

# region >> Reset Param Buttons
@app.callback(
    [
        ddo('pbtn_rescore', 'n_clicks'),
        ddo('pbtn_restore', 'n_clicks'),
        ddo('ddw_chg', 'n_clicks'),
        ddo('ddw_clr', 'n_clicks')
    ],
    [
        ddi('hidden_score', 'children'),
        ddi('hidden_currp', 'children'),
    ]
)
def reset_param_btns(s,c):
    sysout('\n>> [PIPE]: Resetting Button Values ')
    return 0, 0, 0, 0

# endregion < Reset Param Buttons

# region >> Dynamically Update Score Input
@app.callback(
    ddo('ddw_in','value'),
    [
        ddi('ddparamweight','value')
    ],
    [
        dds('hidden_currp','children')
    ]
)
def place_curr_weight(ws,hd):
    if not isok(hd): return 0
    else:
        df = fromjson(hd)
        df.index = df['ID']
        return df.loc[ws]['Weight']

#  endregion << ...

# region >> Update Hidden Storage
@app.callback(
    [
        ddo('hidden_currp', 'children'),
        ddo('hidden_score', 'children'),
        ddo('footer_p','children')
    ],
    [
        ddi('ddstatesort', 'value'),
        ddi('ddparamsort', 'value'),
        ddi('pbtn_rescore', 'n_clicks_timestamp'),
        ddi('pbtn_restore', 'n_clicks_timestamp'),
        ddi('ddw_chg','n_clicks_timestamp'),
        ddi('ddw_clr','n_clicks_timestamp')
    ],
    [
        dds('hidden_currp', 'children'),
        dds('hidden_score', 'children'),
        dds('param_table', 'selected_rows'),
        dds('state_table', 'selected_rows'),
        dds('daterange', 'start_date'),
        dds('daterange', 'end_date'),
        dds('pbtn_rescore', 'n_clicks'),
        dds('pbtn_restore', 'n_clicks'),
        dds('ddw_chg','n_clicks'),
        dds('ddw_clr', 'n_clicks'),
        dds('ddw_in','value'),
        dds('ddparamweight', 'value'),
    ]
)
def updater(srt_s,srt_p,pb1,pb2,pbw1,pbw2,hp,hs,sp,ss,d0,d1,
            pb1_nc,pb2_nc,pbw1_nc,pbw2_nc,w_in,w_tag):
    sysout('\n>> [Updating]\n')
    # Create Base Versions
    dname = [int(x.split('_')[1]) for x in list(dref['id'])]; dtag = helper_dref2tag()
    hp0 = kv2df(
        ['ID', 'Parameter', 'Weight', 'Selected'],
        [
            dname,
            dtag,
            [list(weights['weights'])[x] for x in range(len(dname))],
            [True for x in range(len(dname))]
        ]
    )

    hs0 = kv2df(
            k=['State', 'Rank','Score','Selected'],
            v=[states,
               cscore(score_data0,dates,True,1),
               cscore(score_data0,dates,False,1000000),
               [True for x in states]]
    )

    ftext = 'Covering {0}/50 States and Using {1}/95 Parameters Indicates the\n' \
            'Top 5 States to Live In are: [ {2}, {3}, {4}, {5} and {6} ]'

    if not isok(hp) or not isok(hs) or pb2_nc > 0:
        sysout('\n    + Return Original Value (none or restore pressed)')
        hs0.sort_values(['Score'],inplace=True,ascending=False)
        iso_score = [x for x in list(hs0['State'])[:5]]
        return (
            tojson(df=hp0), tojson(df=hs0),
            ftext.format(len(hs0.index),len(hp0.index),
                iso_score[0],iso_score[1],iso_score[2],iso_score[3],iso_score[4])
        )
    else:
        # [I] Load, isolate elements
        pt = fromjson(hp)
        pt.index = [i for i in range(len(pt.index))]
        if not isok(sp): sp = list(pt.index)
        curr_p = list(pt.loc[sp]['Parameter'])
        curr_w = list(pt.loc[sp]['Weight'])
        curr_id = list(pt.loc[sp]['ID'])

        # Handle weight updates
        new_w =  list(pt['Weight'])
        if pbw2_nc > 0:
            new_w = [0 for x in new_w]
        elif pbw1_nc > 0:
            if isok(w_in):
                print('\n>> [WEIGHT CHANGE]')
                loc = 0
                while not list(pt['ID'])[loc] == w_tag: loc += 1
                new_w[loc] = float(w_in)
            # Reset
            # pt.index = [i for i in range(len(pt.index))]
            # curr_w = list(pt.loc[sp]['Weight'])

        # Check if weights changed, if so, recreate pt
        if not new_w == list(pt['Weight']):
            pt = kv2df(
                ['ID', 'Parameter', 'Weight', 'Selected'],
                [
                    list(pt['ID']),
                    list(pt['Parameter']),
                    new_w,
                    list(pt['Selected'])
                ]
            )
            curr_w = list(pt.loc[sp]['Weight'])

        # Load state table and extract current states
        st = fromjson(hs)
        st.index = [i for i in range(len(st.index))]
        if not isok(ss): ss = list(st.index)
        curr_s = list(st.loc[ss]['State'])

        # print('curr_p: ',curr_p)
        # print('curr_s: ',curr_s)
        # print('Below: curr_p / curr_w / curr_id / curr_s\n')
        # print(curr_p,'\n',curr_w,'\n',curr_id,'\n',curr_s,'\n')

        # [II] Handle rescoring (First make temp curr_score of current results)
        if pb1_nc > 0:
            curr_keys = [mkstr('dataset_',q,' ') for q in curr_id]
            curr_dset = usekeys(dset.copy(), curr_keys)
            curr_score = calcScoreData(curr_dset, unlist(['date', 'dateu', states]), curr_w)

            # Re-create State scores table -- Apply Sorting with dates
            # print(d0,' - ',type(d0),'\n',d1,' - ',type(d1))
            d0 = int(yahoo_calc_date(d0))
            d1 = int(yahoo_calc_date(d1))
            print('\n\nD0: ',d0,' and D1: ',d1,'\n')
            new_dates = range(d0,d1)
            st = kv2df(
                ['State', 'Rank', 'Score', 'Selected'],
                [
                    states,
                    cscore(curr_score, new_dates, True, 1),
                    cscore(curr_score, new_dates, False, 1000000),
                    [bool(list(st['State'])[x] in curr_s) for x in range(50)]
                ]
            )


        # [III] Handle Sorting - Maintains current selections
        if isok(srt_p):
            sysout('\n    + [>] {Param} Sort by: ', srt_p)
            a = srt_p.split('_')[0]; b = srt_p.split('_')[1]
            to_eval = mkstr('pt.sort_values(["', a, '"], ascending= ', b, ') ')
            sysout('\n      + [>]', to_eval, '\n')
            pt = eval(to_eval)
            pt['Selected'] = [x in curr_p for x in list(pt['Parameter'])]

        if isok(srt_s):
            sysout('\n    + [>] {State} Sort by: ', srt_s)
            a = srt_s.split('_')[0]; b = srt_s.split('_')[1]
            to_eval = mkstr('st.sort_values(["', a, '"], ascending= ', b, ') ')
            sysout('\n      + [>]', to_eval, '\n')
            st = eval(to_eval)
            st['Selected'] = [x in curr_s for x in list(st['State'])]

        # [IV] Return to hidden storage for front-end rendering
        st2 = st.copy()
        st2.sort_values(['Score'], inplace=True, ascending=False)
        iso_score = []; i = 0
        while len(iso_score) <= 5 or i < len(st2.index):
            to_try = list(st2['State'])[i]
            if to_try in curr_s: iso_score.append(to_try)
            i += 1

        return (
            tojson(df=pt), tojson(df=st),
            ftext.format(len(ss), len(sp),
                         iso_score[0], iso_score[1], iso_score[2], iso_score[3], iso_score[4])
        )

# endregion << Update Hidden Storage ------


# region >> Update Param Table

# region [I] Update Param Cells Style
@app.callback(
    ddo('param_table', 'style_cell_conditional'),
    [
        ddi('tblp_param', 'children'),
        ddi('param_table', 'selected_rows')
    ],
    [

    ]
)
def style_params(pt,ps):
    if isok(pt) and isok(ps):
        sysout('\n>> [style_params]  ')
        ns = [x for x in range(0, 96) if x not in ps]
        print('\n-----\nps: ', ps, '\n-----')
        print('pns: ', ns, '  \n-----')
        return unlist([
            [{'if': {'row_index': s}, 'background-color': '#9effae'} for s in ps],
            [{'if': {'row_index': s}, 'background-color': '#ffd1d1'} for s in ns],
            [
                {'if': {'column_id': 'ID'}, 'width': '10%'},
                {'if': {'column_id': 'Parameter'}, 'width': '78%','font-size':'12px'},
                {'if': {'column_id': 'Weight'}, 'width': '12%'}
                # {'if': {'column_id': 'Score'}, 'width': '18%'}
            ]
        ])

# endregion <

# region [II] Update Selected Params

@ app.callback(
    ddo('param_table','selected_rows'),
    [
        ddi('pbtn_rmvall', 'n_clicks_timestamp'),
        ddi('pbtn_addall', 'n_clicks_timestamp')
    ],
    [

    ]
)
def chg_params(c,a):
    if isok(c) and isok(a) and cSum([c,a]) > 0:
        sysout('\n>> [chg_params] ')
        if max(c,a) == a:
            ps = [x for x in range(0, 96)]; ns = []
        if max(c, a) == c:
            ps = []; ns = [x for x in range(0, 96)]
        sysout('\n    + [PIPE]: Changing ...')
        return ps
    else: sysout('\n>> [chg_params] FAILED  ')

# endregion

# region [III] Update Param Main Table
@app.callback(
    ddo('tblp_param','children'),
    [
        ddi('hidden_currp','children')
    ]
)
def op_paramtable(hp):
    sysout('\n>> [op_paramtable]  ')
    if not isok(hp):
        wref = list(weights['weights'])
        dname = [int(x.split('_')[1]) for x in list(dref['id'])]
        dtag = helper_dref2tag()

        df = kv2df(
            ['ID', 'Parameter', 'Weight', 'Selected'],
            [
                dname,
                dtag,
                [list(wref)[x] for x in range(len(dname))],
                [True for x in range(len(dname))]
            ]
        )
    else: df = fromjson(hp)

    ps = [x for x in range(len(list(df['Selected']))) if bool(list(df['Selected'])[x]) is True]
    df = df.iloc[:, 0:3]

    tp_st = {
        # 'overflowX': 'scroll',
        'overflowY': 'scroll',
        'maxHeight': '350px',
        'maxWidth': '1000px',
        'border': 'thin black solid'
    }
    tp_sc = {
        'minWidth': '10px',
        'maxWidth': '250px',
        'textOverflow': 'ellipsis',
        'overflow': 'hidden',
        'whiteSpace': 'normal'
    }

    dashtable = dtbl.DataTable(id='param_table',
                               columns=[
                                   {
                                       'name': c,
                                       'id': c,
                                       'type': ifelse(i == 0, 'str', 'numeric'),
                                       'format': ifelse(i > 0,
                                                        Format(nully='na', precision=2),
                                                        Format(nully='na',precision=2))
                                   }
                                   for i, c in enumerate(list(df.columns))
                               ],
                               data=df.to_dict('rows'),
                               editable=False,
                               filtering=False,
                               sorting=False,
                               sorting_type="multi",
                               row_selectable="multi",
                               row_deletable=False,
                               style_table=tp_st,
                               style_cell=tp_sc,
                               style_cell_conditional=[],  # tp_scc,
                               # n_fixed_rows=1,
                               # n_fixed_columns=2,
                               selected_rows=ps,
                               style_as_list_view=True)
    return h.Div(dashtable)


# endregion

# endregion << Update Param Table -----

# region >> Update State Table

# region [I] Update Selelected States
@app.callback(
    ddo('state_table','selected_rows'),
    [
        ddi('sbtn_rmvall', 'n_clicks_timestamp'),
        ddi('sbtn_addall', 'n_clicks_timestamp')
    ]
)
def chg_states(c,a):
    if isok(c) and isok(a) and cSum([c,a]) > 0:
        sysout('\n>> [chg_states] ')
        if max(c,a) == a:
            ss = [x for x in range(0, 51)]; ns = []
        if max(c, a) == c:
            ss = []; ns = [x for x in range(0, 51)]
        sysout('\n   + [PIPE]: Changing ... ')
        return ss
    else: sysout('\n>> [chg_states] FAILED ')

# endregion [I]

# region [II] Apply Styling to selected states (rows)
@app.callback(
    ddo('state_table','style_cell_conditional'),
    [
        ddi('tblp_state','children'),
        ddi('state_table','selected_rows')
    ]
)
def style_states(st,ss):
    if isok(st) and isok(ss):
        sysout('\n>> [style_states] ')
        ns = [x for x in range(0,51) if x not in ss]
        print('\n-----\nss: ',ss,'\n-----')
        print('ns: ',ns,'  \n-----\n')
        return unlist([
            [{'if': {'row_index': s}, 'background-color': '#9effae'} for s in ss],
            [{'if': {'row_index': s}, 'background-color': '#ffd1d1'} for s in ns],
            [
                {'if': {'column_id': 'State'}, 'width': '50%'},
                {'if': {'column_id': 'Rank'}, 'width': '24%'},
                {'if': {'column_id': 'Score'}, 'width': '24%'}
            ]
        ])

# endregion [II]

# region [III] Update the state table
@app.callback(
    ddo('tblp_state','children'),
    [
        ddi('hidden_score','children')
    ]
)
def op_statetable(hd):
    sysout('\n>> [op_statetable]  ')
    if not isok(hd):
        df = kv2df(
            k=['State', 'Rank','Score','Selected'],
            v=[states,
               cscore(score_data0,dates,True,1),
               cscore(score_data0,dates,False,1000000),
               [True for x in states]]
        )
    else:
        df = fromjson(hd)
        # df['Score'] = [x/1000000 for x in list(df['Score'])]

    ss = [x for x in range(50) if bool(list(df['Selected'])[x]) is True]
    df = df.iloc[:,0:3]

    # print(df)
    # if isok(dsv):
    #     a = dsv.split('_')[0]; b = dsv.split('_')[1]
    #     to_eval = mkstr('df.sort_values(["',a,'"], ascending=',b,')')
    #     df = eval(to_eval)
    # print(df)

    tp_st = {
        # 'overflowX': 'scroll',
        'overflowY': 'scroll',
        'maxHeight': '350px',
        'maxWidth': '1000px',
        'border': 'thin black solid'
    }
    tp_sc = {
       'minWidth': '10px',
       'maxWidth': '50px',
       'textOverflow': 'ellipsis',
       'overflow': 'hidden',
       'whiteSpace': 'no-wrap'
    }

    dashtable = dtbl.DataTable(id='state_table',
                               columns=[
                                   {
                                       'name': c,
                                       'id': c,
                                       'type': ifelse(i == 0, 'str', 'numeric'),
                                       'format': ifelse(i > 0,
                                                        Format(nully='na', precision=2),
                                                        Format(nully='na'))
                                   }
                                   for i, c in enumerate(list(df.columns))
                               ],
                               data=df.to_dict('rows'),
                               editable=False,
                               filtering=False,
                               sorting=False,
                               sorting_type="multi",
                               row_selectable="multi",
                               row_deletable=False,
                               style_table=tp_st,
                               style_cell=tp_sc,
                               style_cell_conditional=[],#tp_scc,
                               # n_fixed_rows=1,
                               # n_fixed_columns=2,
                               selected_rows=ss,
                               style_as_list_view=True
    )
    return h.Div(dashtable)

# endregion < [III]

# endregion << Update State Table -----

# region >> Update Map
@app.callback(
    ddo('state_map','figure'),
    [
        ddi('state_table','selected_rows'),
        ddi('hidden_score','children')
    ]
)
def op_map(ss,hd): #,br,bs):
    # Extract Data (Get source if everything else is empty
    if isok(hd):
        sysout('\n>> [MAP]: Loading Current Data')
        df = fromjson(hd)
        # df['Score'] = [x / 1000000 for x in list(df['Score'])]
    else:
        sysout('\n>> [MAP]: Loading original Data')
        df = kv2df(
            k=['State', 'Rank', 'Score'],
            v=[states,
               cscore(score_data0, dates, True, 1),
               cscore(score_data0, dates, False, 1000000)]
        )
        ss = [i for i in df.index]

    print('\nIN THE MAP\n>SS:\n')
    print(ss)


    # Handle Map Geo Locations from state table
    df.index = [i for i in range(len(df.index))]
    stag = list(df.loc[ss]['State'])
    print('\n>STAG:\n',stag)
    loc = mkdict([x.lower() for x in list(us_dict['state_name'])], [x for x in list(us_dict['state_abbr'])])
    s_abbr = [loc[x] for x in stag]
    new_df = kv2df(
        k=['State','abbr','Rank0','Score'],
        v=
        [
            stag,
            list(s_abbr),
            list(df.loc[ss]['Rank']),
            # [str(i+1) for i in range(len(stag))],
            list(df.loc[ss]['Score'])
        ]
    )

    map_tag = 'State Scores (weighted and scaled by 1mil)'

    # Create subset ranks (if states selected < 50)
    # if len(list(df['State'])) == 50: subset_data = ['--' for i in range(len(df['State']))]
    # else: subset_data = [str(i+1) for i in range(len(df['State']))]


    # Create Hover Text
    # maptext = list(
    #     '> '+new_df['State']+' ('+new_df['abbr']+')'+'<br>'+
    #     '    + '+'Overall Rank|Score    : '+new_df['Rank0']+' | '+new_df['Score']+'<br>')#+
    #     '    + '+'Current Rank             : '+new_df['Rank1']+'<br>'
    # )

    # Create Map Data
    mapdata = go.Choropleth(
            colorscale='Jet', #'YlGnBu',
            autocolorscale=False,
            locations=new_df['abbr'],
            z=new_df['Score'],
            # zmin=min(df['avg']),
            # zmax=max(df['avg']),
            locationmode='USA-states',
            # text=maptext,
            # text = df['state_name'],
            marker={'line': {'color': 'rgb(0,0,0)','width': 1.0}},
            colorbar={"thickness": 15,"xpad":2,
                      "tickfont":{"size":20,"color":"black"},
                      "tickwidth":15,
                      "len":1,
                      "x": 0.9,
                      "y": 0.5,
                      'title': {"text": '', "side": "bottom"}}
        )

    # Create (Set) Map Design
    design = go.Layout(
        clickmode='event+select',
        dragmode=False,
        paper_bgcolor='rgb(80,100,120)',
        plot_bgcolor='rgb(80,100,120)',
        font={'size':10},
        title={'text':map_tag,'font':{'size':15,'color':'black'}},
        # maxheight=600,
        height=500,
        # maxwidth=600,
        width=600,
        margin={'l':15,'r':10,'t':75,'b':10},
        modebar={'orientation':'h','bgcolor':'black',
                 'color':'gray','activecolor':'white'},
        geo={'showframe': False,
             'showcoastlines':True,
             'coastlinewidth':10,
             'showocean':False,
             "showlakes":False,
             "framecolor":'black',
             "framewidth":10,
             "bgcolor":'rgb(80,100,120)',
             'projection': {'type': "albers usa"}}
    )
    return {"data": [mapdata],"layout": design}


# endregion

if __name__ == '__main__':
    app.run_server(debug=True)