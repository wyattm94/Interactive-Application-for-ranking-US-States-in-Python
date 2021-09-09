'''
[Quick Format]
> File: app_utilitiles.py (yes, utilitiLes.py, there is an L, sorry)
> Note: Helpers and wrappers for app2.py (v1-1-0) and current final
'''

import numpy as np
import pandas as pd

import multiprocessing
from itertools import product
from multiprocessing import Process, Queue
import functools as funky

import plotly
import plotly.plotly as ppy
from plotly import graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as h
import dash_daq as daq
import dash_renderer
import dash_table as dtbl
from dash_table.Format import Format, Scheme, Sign, Symbol

from StatsCore import *
from fred_api import *

# region > App (Assignment 4) - reused in final team app --> Integrate into Core

# Functions for session data storage in the application
def tojson(cn=None,cv=None,df=None):
    if isok(df) and isinstance(df,pd.DataFrame):
        return df.to_json(date_format='iso',orient='split')
    else:
        return pd.DataFrame(dict(zip(cn,cv))).to_json(date_format='iso',orient='split')

def fromjson(jd):
    return pd.read_json(jd, orient='split')

# Function to check param types within a structure
def check_inner_types(d,types=None):
    if isok(types): return [isinstance(x, types) for x in d]
    else:
        types = [int,float,str,bool,tuple,list,dict]
        types_s = [str(x).split("'")[1] for x in types]
        v = [[isinstance(x,t) for x in d] for t in types]
        return dict(zip(types_s,v))

# Quick rouding wrapper for structures
def roundwrap(x,sigfig=4):
    if isinstance(x,list):
        return [round(a,sigfig) for a in x]
    elif isinstance(x,dict):
        return 1
    elif isinstance(x,pd.DataFrame):
        return 1
    else: return round(x,sigfig)

# Create html (core) table from pandas Datatframe
def mktbl(df):
    a = h.Table(
        [h.Tr([h.Th(cn) for cn in df.columns])]
        +
        [h.Tr([h.Td(df.iloc[i, j]) for j,cn in enumerate(df.columns)]) for i in range(len(df.index))])
    sysout('\nTable Made:\n')
    # p2log(a)
    return(a)

# endregion

# region :: Class - Fred + Associated Helpers/Runners

# Handles data loading from source (reading)
def fredloader():
    gc.collect()
    sysout('\n[>>] Fetching Data Sets [Time: ',showdt(),'] ...'); t0 = time.time()

    dpaths = ['fred_warehouse/fred_app_sourcedata.xlsx',
                    'fred_warehouse/fred_app_sourcedata_ref.xlsx',
                    'fred_warehouse/us_states.xlsx',
                    'fred_warehouse/fred_app_weights0.xlsx']

    with mtpool(processes=len(dpaths)) as pool:
        results = pool.starmap(xb2py, product(dpaths))
    pool.terminate(); gc.collect()
    dset,dref,us_data,weights = (x for x in results)

    # Fix (temp) broken Data and apply index (posix unix timestamps - dateu)
    temp = dict(); fullindex = None
    for i,(k, v) in enumerate(dset.items()):
        if i==0: fullindex = list(v['dateu'])
        if k in ['dataset_2', 'dataset_6', 'dataset_11', 'dataset_31']: continue
        else:
            v.index = fullindex; temp[k] = v
    dset = temp.copy()

    # del temp, fullindex,i,k,v,results,dpaths

    us_dict = us_data.to_dict('list')
    states = [s.lower() for s in us_data['state_name']]; tchg = roundwrap(time.time() - t0,3)
    sysout('\n[OK] Finished in {t:6.3f}'.format(t=tchg),' sec. [Time: ',showdt(),']')
    return dset,dref,us_dict,states,weights

# Handles Initial weight calculations
def calcweights(db,wmin=0.10,wmax=0.95,umod=None):
    sysout('\n[>>] (fred) Calculate Starting Weights')
    cn = ['dataset', 'stddev']; dbkeys = list(db.keys())
    d0 = [v.iloc[rng(0, 145, 12, drop_pre0=False),] for k, v in db.items()]

    sysout('\n[>>] MultiProcess: Calculate yearly changes {for weights} ')
    t0 = time.time()
    d1 = None
    with multiprocessing.Pool(processes=10) as pool:
        d1 = pool.starmap(calc_return2, product(d0))
        # pool.close()
        # pool.join()
    pool.terminate();
    gc.collect()
    sysout('\n[OK] TS-Weight data resource  ')
    sysout('\nTime (in seconds): ',round(time.time()-t0,3))

    d2 = [aggregate(x.iloc[:, 2:]) for x in d1]
    d3 = [np.std(list(x)) for x in d2]
    wd = kv2df(cn, [dbkeys, d3])
    wd.sort_values(['stddev'], ascending=False, inplace=True)

    # Validate numerics (non-NaN) - backup if constructor filter fails - should have no effect
    wstd = [float(x) for x in wd['stddev'] if isinstance(x, type(np.nan))]
    wrnk = rng(1, len(wd.index) + 1, 1, r=3)  # Ranks

    ww = [x for x in rng(wmin, wmax, j=len(wd.index), r=3)]
    return cbind(wd, mkdict(['w_rank', 'weights'], [wrnk, ww]))

# Handles score/rank calculations
def calcScoreData(db,cid,w):
    sysout('\n[>>] MProcess: Score Data Extract  ')
    t0 = time.time()
    # print(list(db.keys())[0])
    # print(db[list(db.keys())[0]])
    datecols = [list(db[list(db.keys())[0]][x]) for x in ['date','dateu']]
    temp = [[] for i in cid[2:]]
    if istype(w,pd.DataFrame):
        w.index = [x for x in list(w['dataset'])]
        w = w.loc[list(db.keys())]
        w = list(w['weights'])

    for j,c in enumerate(cid[2:]):
        for i,(k,v) in enumerate(db.items()):
            v.columns = [x.lower() for x in list(v.columns)]
            v = v.iloc[:,2:]
            if i==0: temp[j] = [float(x)*w[i] for x in list(v[c])]
            else:
                temp[j] = addlists(temp[j],[float(x)*w[i] for x in list(v[c])])

    finalv = datecols + temp
    score = kv2df(cid,finalv)
    finalranks = [[] for i in range(len(cid[2:]))]
    for i in list(score.index):
        temp = list(score.loc[i])[2:]
        temp = kv2df(['state', 'score'], [cid[2:], temp])
        temp.sort_values(['score'], ascending=False, inplace=True)
        ranks = [i for i in range(1, 51)]
        temp = kv2df(['state', 'score', 'rank'],
                     [list(temp['state']), list(temp['score']), ranks])
        temp.sort_values(['state'], ascending=True, inplace=True)
        for j, x in enumerate(temp['rank']):
            finalranks[j].append(x)

    finalv = datecols + finalranks
    ranks = kv2df(cid,finalv)
    sysout('\n[TIME] (in seconds): ', round(time.time() - t0, 3))
    return dict(
        score=score,
        rank=ranks
    )

# Define Runner-Mains
def calc_return2(d,chg=False,sigfig=2,c_iso=[(0,2)],norm=False,prnt=False):
    if prnt: sysout('\n[>>] Calculating Returns ... ')
    if not istype(d,pd.DataFrame,list): return end('[ERROR] Bad input type')
    elif istype(d,pd.DataFrame):
        if prnt: sysout('[Input was DF] ')
        dc_hold = None; orig_cn = []
        if isok(c_iso):
            orig_cn = slicer(c_iso,list(d.columns),True)
            dc_hold = d[orig_cn]
            d.drop(orig_cn,axis=1,inplace=True)

        if chg:
            return [calc_return(mklist(d[c]),chg=True,sigfig=sigfig,c_iso=c_iso,norm=norm,prnt=False)
                    for c in list(d.columns)]
        cn   = list(d.columns)
        out = [calc_return(mklist(d[c]),chg=chg,sigfig=sigfig,c_iso=c_iso,norm=norm,prnt=False)
               for c in list(d.columns)]

        # If withheld, add bad in the beginning (usually date/index isolated)
        if isok(dc_hold):
            cn = orig_cn + cn
            if norm: out = [list(dc_hold[x]) for x in orig_cn ] + out
            else: out = [list(dc_hold[x].iloc[1:]) for x in orig_cn ] + out
        return kv2df(cn,out)
    elif istype(d,list):
        if len(d) <= 1: return d
        elif chg: return roundwrap((d[(-1)]-d[0])/d[0],sigfig)
        elif norm: return [0]+[roundwrap((d[i] - d[0]) / d[0], sigfig) for i in range(1, len(d))]
        else: return [roundwrap((d[i]-d[(i-1)])/d[(i-1)],sigfig) for i in range(1,len(d))]
    else: return end('[ERROR] Unknown param type d')

    # Helpers

# Define Runners
def calc_return3(d): return calc_return2(d,True)
def calc_return4(d): return calc_return2(d,norm=True)

# Other helpers
def addlists(a,b):
    a = [float(x) for x in a]
    b = [float(x) for i,x in enumerate(b) if i < len(a)]
    return [(a[i]+b[i]) for i in range(len(a))]
def rowsums(df,c_iso=[(0,2)]):
    if isok(c_iso):
        dates = slicer(c_iso,df,bycol=True); df = df.iloc[:,2:]
    else: dates = None
    sums = [cSum([x for x in list(df.loc[k])]) for k in list(df.index)]
    return sums

class Fred:

    # STABLE
    # region > All member variables (Public and Private)
    __flagStart,__flagUpdate = False,False
    __maxworkers=15; __timer0, __timer1 = None,None
    __timeseries = (None for i in rng(2)); __dbscore = None
    db,keys,weights,scores, ranks,ref,cols,rows,tState,tParam = (None for i in range(10))

    # endregion

    # STABLE
    def __init__(self,d=None,r=None,w=None,dsa=None,run0=False):
        sysout('\n[>>] Creating fred object ... ')
        if not isok(d) or not isok(r) or not isok(w) or not isok(dsa):
            sysout('\n[ERROR] Member d is None (no input param)')
        else:
            self.timer0 = time.time()
            self.db = d
            self.keys = list(self.db.keys())

            w = w.sort_values(['dataset'],ascending=True)
            self.weights = dict(reset=w,curr=w)
            self.ref = r
            self.__dbscore = dsa

            self.cols = list(self.db[self.keys[0]].columns)
            self.rows = list(self.db[self.keys[0]].index)
            self.states = self.cols
            self.dates = self.rows

            self.pipeParam()
            self.pipeTime()
            self.pipeState()
            self.pipeTimeseries()

            sysout('\n[FRED] Initiliazed (seconds) :',round(time.time()-self.timer0,3))

    # STABLE
    # region > Handlers (row/col dim, keys (datasets/params) and timeseries
    '''> ... '''
    def __dimhandle__(self,c=None,r=None):
        sysout('\n[>>] (fred) Handling column and row adjustments')
        ucn,ucs,urn,urs = (None for i in range(4))

        # Deal with columns
        if isok(c) and istype(c,list,int,float,str):
            if not istype(c,list): c = [c]
            c = [int(x) if istype(x,int,float) else str(x) for x in c]
            if all([istype(x,int) for x in c]): ucn = [self.__cols[x] for x in c]
            else: ucs = c
        else:
            sysout('\n[NOTE] Bad c value (none = ALL, other (error) = ALL)')
            updcols = self.states

        # Deal with rows
        if isok(r) and istype(r,list,int,float,str):
            if not istype(r, list): r = [r]
            r = [int(x) if istype(x, int, float) else str(x) for x in r]
            if all([istype(x, int) for x in r]): urn = [self.dates[x] for x in r]
            else: urs = r
        else:
            sysout('\n[NOTE] Bad r value (none = ALL, other (error) = ALL)')
            updr = self.dates
        temp = usekeys(self.db,self.keys); holder = dict()

        # Determine proper type, adjust accordingly
        for k,v in temp.items():
            if isok(ucn): v = v[ucn]
            else: v = slicer(ucs,v,bycol=True)
            if isok(urn): v = v.loc[urn]
            else: v = slicer(urs,v)
            holder[k] = v

        self.db = holder.copy(); del holder
        self.__update__()
        sysout('\n[OK] DB Updated')



    # STABLE
    def __handleTseries__(self,t=None,run0=False):
        ds,p,s,g = 0,0,0,0
        sysout('\n[>>] (fred) Calculting timeseries:')
        # if not isok(self.__timeseries) or not isok(t):
        if run0:
            ds = [v for k, v in self.db.items()]

            if run0 or not isok(self.__timeseries):
                t0=time.time()
                sysout('\n[+>] Calculating time series: Periodic Change  ')
                with multiprocessing.Pool(processes=self.__maxworkers) as pool:
                    p = pool.starmap(calc_return2, product(ds))
                pool.terminate()
                gc.collect()
                sysout('\n[OK] TS-Periodic Done')
                sysout('\n[TIME] (in seconds): ', round(time.time() - t0, 3))
            else:
                p = self.__timeseries['p']

            sysout('\n[+>] Calculating time series: Full Period Change (single)  ')
            t0=time.time()
            with multiprocessing.Pool(processes=self.__maxworkers) as pool:
                s = pool.starmap(calc_return3, product(ds))
            pool.terminate()
            gc.collect()
            sysout('\n[OK] TS-1pChange Done')
            sysout('\n[TIME] (in seconds): ', round(time.time() - t0, 3))

            sysout('\n[+>] Calculating time series: Normalized Growth (from base 0)  ')
            t0=time.time()
            with multiprocessing.Pool(processes=self.__maxworkers) as pool:
                g = pool.starmap(calc_return4, product(ds))
            pool.terminate()
            gc.collect()
            sysout('\n[OK] TS-Growth Done')
            sysout('\n[TIME] (in seconds): ', round(time.time() - t0, 3))

            self.__timeseries = mkdict(['p','s','g'],[p,s,g])

        if isok(t):
            to_eval = mkstr('self.__timeseries[',t,']'); return eval(to_eval)
        else: return 0

    # endregion

    # TODO: Make the filtering functionality
    def __dofilter__(self,*kwargs):
        print('WIP')
    def __doscore__(self,umod=None,update=False):
        if not isok(umod): umod = self.weights['curr']

        self.__dbscore = calcScoreData(self.db,self.states,umod)
    def __getScores__(self):
        s = self.__dbscore['score'].loc[self.rows,self.cols]
        r =self.__dbscore['rank'].loc[self.rows,self.cols]
        return dict(s=s,r=r)

    #  Restores all data to current
    def restore(self):
        sysout('\n[<<] {Restoring Data}')
        self.__restore__()

    # Assumes search using 'dataset_x' as key (tag2id as True reverses search)
    def maprefid(self,n,tag2id=False):
        sysout('\n[<<] {ID Search}')
        if not isinstance(n,list): n = [n]
        sfrom,sto = ifelse(bool(tag2id),
                           (self.__ref['tag'],self.__ref['id']),
                           (self.__ref['id'],self.__ref['tag']))
        # sfrom = ifelse(bool(tag2id), self.__ref['tag'], self.__ref['id'])
        return usekeys(mkdict(sfrom,sto),n)

    # For State Table ops (Columns)
    def calc_tState(self,sub=None):
        if isok(sub):
            print(1)

        self.tState = kv2df(
            k=['State', 'Rank', 'Score', 'Selected'],
            v=[self.states[2:], self.__cscore__(), self.__crank__(),
               [True for i in self.states[2:]]]
        )
    def __cscore__(self):
        temp = self.__dbscore['score']
        temp.index = temp['dateu']
        sub = temp.loc[self.rows]
        return [cSum(list(sub[x]))/len(sub.index) for x in self.states[2:]]
    def __crank__(self):
        temp = self.__dbscore['rank']
        temp.index = temp['dateu']
        sub = temp.loc[self.rows]
        return [cSum(list(sub[x]))/len(sub.index) for x in self.states[2:]]
    def pipeState(self,din=None):
        if isok(din):
            self.cols = list(din['State'])
            self.tState['Selected'] = [x in self.cols for x in list(self.tState['State'])]
            return self.tState
        else: return self.tState
    def change_weights(self, umod=None):
        if not isok(umod):
            return end('[ERROR] No weights input for change_weights')
        else:
            if not istype(umod, list) or len(umod) < len(self.__weights.index):
                return end('[ERROR] Bad input type/length < # total datasets')
            else:
                self.weights['curr']['weights'] = umod
                self.__score__()

    # For Param Table ops (keys and weights)
    def __keyhandle__(self,k): self.db = usekeys(self.__db,k,True); self.__update__()
    def __cweights__(self,din=None,dget=None):
        w = self.weights['curr']; w.index = w['dataset']
        wfix = w.loc[self.rows]; print(wfix)
    def pipeParam(self,din=None):
        if self.__flagStart:
            self.tParam = kv2df(
                k=['State','Weight','Selected'],
                v=[self.states[2:],self.change_weights(),
                   [True for i in self.states[2:]]]
            )
    def pipeTime(self,din=None):
        if isok(din):
            self.rows = sorted([x in self.dates for x in din])
            self.__doscore__()
    def pipeTimeseries(self,din=None,dout=None):
        print('a')

# endregion