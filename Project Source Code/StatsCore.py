'''
_____________________________________________________
> File: StatsCore.py
> Root: C:/finapp/root_app
> Assets: None
> External: None
> Note:
    + Core and advanced stats equations
    + Related numerical tools/utilities
    + [WIP] StatsCore class object
> Author: Wyatt Marciniak
_____________________________________________________
'''


# region :: Dependencies
# Non-specific modules (general use, IO, system, OS, etc...)
import os, gc, sys, math, time, certifi, xlsxwriter
import random as rand
import datetime as dt
from datetime import datetime as dtt

# More focused data structure modules
import numpy as np
import pandas as pd
import openpyxl as opx

# Web based interactions (web-scrapers, spiders, iso-requests,etc...)
import requests as req
from bs4 import BeautifulSoup

# Multi-threading utilities
import threading
from threading import Thread as mt
from threading import active_count as nlive

# Multi-processing utilties (extends a lot of core threading funtionality)
import multiprocessing
from multiprocessing.pool import ThreadPool as mtpool
from multiprocessing import Process, Queue

# Custom Scripts
from helper import *

# endregion

# region :: Base calculations and moments
'''
[>] cMu(): 
[>] cVar():
[>] cStd():
[>] cSkew():
[>] cKurt():
[>] pqqtiles():
'''

def cMu(x): return sum(x)/len(x)
def cVar(x):
    vmu = cMu(x); var = 0
    for e in x: var += (e-vmu)**2
    return var/len(x)
def cStd(x): return math.sqrt(cVar(x))
def cSkew(x): return None
def cKurt(x): return None
def pqqtiles(d,t,run='percentile',r=None):
    if not isinstance(d,list): d = [d]; d = sorted(d)
    if isinstance(t,(int,float)): t = rng(0,1.01,j=int(t))
    loc = [len(d) * t[i] for i in range(len(t))]; val = []

    # if len(loc) == 1: return round(d[int(math.ceil(loc[0]))],ifelse(bool(isok(r)),r,2))

    for i, x in enumerate(loc):

        if x % 1 == 0 and i > len(loc):
            print(x)
            val.append(cSum([d[x], d[(i+1)]]) / 2)
        else:
            val.append(d[min(int(math.ceil(x)),(len(d)-1))])
    if isok(r):
        val = [round(x,r) for x in val]
    return val

# endregion



# region :: Base summation/product/lower-level iterable tools
'''
[>] cSum(): 
[>] cProd():
[>] blinder():
'''

def cSum(x):
    csum = 0
    for a in x:
        csum += a
    return csum
    # return eval(mkstr(x,sep_list='+'))
def cProd(x): return eval(mkstr(x,sep_list='*'))
def blinder(x,b): return eval(mkstr(x,sep_list=b))

cSum([1,2,3])

# endregion

# region :: Modified funtionalities for creating and using sequence-structures
'''
[>] rng(): 
[>] sample():
[>] unique():
'''

def rng(s=None,e=None,b=None,j=None,r=None,drop_pre0=True):
    try:
        if isok(s) and not isok(e): return rng(0,s,b,j)
        else:
            out = list(); try_next = s; i = 0
            while try_next < e:
                if isok(j) and i == j: break
                else:
                    out.append(ifelse(bool(isok(r)),round(try_next,r),try_next))
                    i += 1
                if isok(b): try_next += b
                elif not isok(b) and isok(j): try_next += (e-s)/j
                else: try_next += 1
            if drop_pre0 and out[0] == 0: return out[1:]
            else: return out
    except IOError: return end('[ERROR] Bad Inputs {rng()}')
    except TypeError: return end('[ERROR] Bad inputs to output maker {rng()}')
def sample(seq,n,replace=True,duplicates=True):
    try:
        if not isinstance(seq,list) or not isinstance(n,int): raise IOError
        if not replace and n > len(seq): raise IndexError
        out = []
        if not duplicates: seq = unique(seq)
        for i in range(n):
            x = rand.randint(0,len(seq)-1); out.append(seq[x])
            if not replace: seq.remove(seq[x])
        return ifelse(len(out)==1,out[0],out)
    except IOError:
        sysout('\n[ERROR] Bad input (seq = list, n = int, replace = boolean)')
    except IndexError:
        sysout('\n[!] n > length of seq with replace = False (Returned everything)')
        return seq
def unique(x):
    '''
    :param x: list of strings
    :return: list of unique elements in x
    '''
    out = []
    for e in x:
        if not e in out: out.append(e)
    return out

# endregion



# region :: Time-series, change (%), etc.... functionalities
'''
[>] calc_return()
'''
# sad = calc_return(d0[1],norm=True)
# sad
# len(d0[0].columns)

def calc_return(d,chg=False,sigfig=4,c_iso=None,norm=False,prnt=False):
    if prnt: sysout('\n[>>] Calculating Returns ... ')
    if not istype(d,pd.DataFrame,list): return end('[ERROR] Bad input type')
    elif istype(d,pd.DataFrame):
        if prnt: sysout('[Input was DF] ')
        dc_hold = None; orig_cn = []
        if isok(c_iso):
            if istype(c_iso,list):
                orig_cn = []
                for x in c_iso:
                    if istype(x,int,float):
                        x = int(x)
                        if list(d.columns)[x] not in orig_cn: orig_cn.append(list(d.columns)[x])
                    elif istype(x,str):
                        if x not in orig_cn: orig_cn.append(x)
                    else:
                        # Assumes tuple or list (anything else will break - trys coming later)
                        oc_range = [a for a in list(d.columns) if a in list(range(x[0],x[1])) and a not in orig_cn]
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


# endregion
