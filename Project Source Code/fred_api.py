# -*- coding: utf-8 -*-
"""
__________________________________________________________________________________________________

    > Script ID     : fred_api.py
    > Author(s)     : Wyatt Marciniak [wyattm94@gmail.com]
    > Maintainer(s) : Wyatt Marciniak
    > License(s)    : Copyright (C) Wyatt Marciniak - All Rights Reserved [See LICENSE File]
    > Description   : Helper Functions (assoc. web scraping APIs) - Specifically Designed For the FRED Database
__________________________________________________________________________________________________
<<< Source Code Begins >>>
"""

## ----- Imports
import os
import numpy as np
from helper import *
from StatsCore import *
from yahoofinance import *
from stockinfo import *
import certifi

# region :: Core (to mege with base)

## Drop rows
def rdrop(df,rows=None,by={},keep=False):
    if isinstance(df,dict):
        out = dict(); keys = list(df.keys())
        for k in keys: out[k] = rdrop(df[k],rows,by,keep)
        return out
    else:
        # Cols are key:val pairs > Single DF (broken into dict)
        data = df.to_dict(); outd = dict()
        keys = list(data.keys())
        for k in keys:
            temp = data[k] # Col in DF (Iterate by cols)
            if isok(rows): rows = rows
            else:
                loc_min = int(np.min([int(x) for x in list(np.where(df[by['target']] >= by['min'])[0])]))
                loc_max = int(np.min([int(x) for x in list(np.where(df[by['target']] >= by['max'])[0])]))
                rows = list(range(int(loc_min),(int(loc_max)+1),1))

            if keep: outd[k] = [temp[i] for i in range(len(temp)) if i in rows]
            else: outd[k] = [temp[i] for i in range(len(temp)) if i not in rows]
        return pd.DataFrame(outd)


    aaq3 = rdrop(da2,loc)

## Annual (1) to monthly (12) data point converter and helper (top)
def ann2mon(d):
    temp = []
    for i in range(0,(len(d)-1)):
        step = float((d[(i+1)] - d[i])/12)
        # print(step)
        temp += [d[i]+(step*s) for s in range(12)]
        # print(temp)
    temp.append(d[-1])
    return temp

def annual2monthly(d,cstart=None,rstart=0,c2add=None,addcol='pre'):
    if isinstance(d,dict):
        out = dict(); keys = list(d.keys())
        for i,k in enumerate(keys):
            print(i,' - ',k)
            out[k] = annual2monthly(d[k],cstart,rstart,c2add[i],addcol)
        return out
    elif isinstance(d,pd.DataFrame):
        if isok(c2add) and addcol=='pre':
            outk = list(c2add.keys())
            outv = [list(c2add[k]) for k in outk]
            # for x in outv:
            #     print('pre-add: ',len(x))
        else: outk, outv = [],[]

        dcols = ifelse(bool(isok(cstart)),
                       list(d.columns)[cstart:],
                       list(d.columns))
        for cn in dcols:
            outk.append(cn)
            # print(outk)
            cd = list(d[cn])[rstart:]
            # print(len(cd))
            # print(len(ann2mon(cd)))
            outv.append(ann2mon(cd))

        if isok(c2add) and addcol == 'post':
            outk += list(c2add.keys())
            outv += [c2add[k] for k in outk]

        return kv2df(outk,outv)
    else:
        sysout('\n[ERROR] Bad Input Type (Must be dict of DF or DF)')
        return None

# Remove duplicate key-named entries
def rmv_state_dups(a,b,aref,bref):
    ka = list(a.keys()); kb = list(b.keys())
    conn_a = open(mkstr(aref,'.txt'),'r'); conn_b = open(mkstr(bref, '.txt'), 'r')

    ra = [x.strip('\n') for x in conn_a.readlines()]; conn_a.close()
    ra = mkdict([x.split('_<(jr)>_')[1] for x in ra],[x.split('_<(jr)>_')[0] for x in ra])
    map_a = [v for k,v in ra.items()]

    rb = [x.strip('\n') for x in conn_b.readlines()]; conn_b.close()
    rb = mkdict([x.split('_<(jr)>_')[1] for x in rb],[x.split('_<(jr)>_')[0] for x in rb])
    map_b = [v for k, v in rb.items()]

    bdups = isin(map_b,map_a,False)['id']
    kb = [x for i,x in enumerate(kb) if i not in bdups]
    return mkdict(kb,[b[k] for k in kb])
    # return dict(ka=ka,kb=kb,ra=ra,rb=rb)

# Rewrite keys of a dictionary
def newkeys(d,newtags):
    temp = d.copy(); d = dict()
    for i, k in enumerate(list(temp.keys())): d[newtags[i]] = temp[k]
    return d

# endregion

# Get coordinates fot the center of the US States
def get_state_coord(write=True,local=False):
    u = 'https://inkplant.com/code/state-latitudes-longitudes'
    raw = wbget(u, 'select', None, 'table')[0]
    head = [s.lower() for s in raw.findAll('tr')[0].strings]
    body = [[] for i in range(3)]
    temp = [list(e) for e in [s.strings for s in raw.findAll('tr')[1:]]]
    for e in temp:
        for i in range(3):
            if i == 0:
                x = str(e[i])
                if x == 'District of Columbia': break
            else: x = float(e[i])
            body[i].append(x)

    df = pd.DataFrame(mkdict(['state_name','coord_lat','coord_lon'], body)).sort_values(['state_name'])
    if write:
        df. to_excel('us_states_coord.xlsx',sheet_name='us_states_coord', index=False)
    if local: return df

# Get US State Abbreviations (bind with name,country and get_state_coord()
def fetch_us_states(d='fred_warehouse'):
    # .xlsx USA, state, state_abbr (all 50)
    writer = pd.ExcelWriter(mkfpath(d=d,f='us_states',e='.xlsx'))
    ur = wbget('https://www.50states.com/abbreviations.htm','select',None,'table')[0]
    # head = [str(x).replace('th>','').strip('<').split(':')[0]
    #         for x in ur.findAll('th')[:2]]
    body    = [x.get_text() for x in ur.findAll('td')[:100]]
    states  = [body[i] for i in range(0,100) if i%2 == 0]
    abbrev  = [body[i] for i in range(0,100) if i%2 != 0]
    s_df    = pd.DataFrame(mkdict(['country','state_name','state_abbr'],
                     [['USA' for i in range(len(states))],states,abbrev]))
    s_df = cbind(s_df,get_state_coord(False,True).to_dict('list'))
    s_df.to_excel(writer,'us_states',index=False)
    writer.save(); writer.close()

## ----- Search FRED (search text, add tags, add a filter, order and sort by params + local/di
def fred_search(txt=None,tag=None,fvar=None,fval=None,ob='popularity',so='desc',
                apikey=None,fdir=None,file='fred_search',sn=dt_tag(),local=False):

    # Step 1: Verify API Key + Parse API Call
    if apikey is None:
        sysout('\n(!) Error: No API key supplied (free on FRED website) - required\n')
        return None
    if txt is None:
        sysout('\n(!) Error: No Text Entries for Searching - required\n')
        return None

    base = 'https://api.stlouisfed.org/fred/series/search?search_text='

    fpath =  ''
    # return(if_(fdir is not None, str(fdir + '/'), fpath))
    if fdir is not None: fpath = str(fdir + '/')
    if file is not None: fpath = str(fpath+ file + '.xlsx')

    if not isinstance(txt,(list,)): base = str(base+txt)
    else: base = str(base+concat_list([l_(x.replace(' ','%20')) for x in txt],sep='+'))


    if tag is not None:
        base = str(base+'&tag_names=')
        if not isinstance(tag,(list,)): base = str(base+tag)
        else: base = str(base+concat_list([l_(x.replace(' ','%20')) for x in tag],';'))

    if fvar is not None and fval is not None: base = str(base+
                                             '&filter_variable='+l_(fvar.replace(' ','%20'))+
                                             '&filter_value='+fval.replace(' ','%20'))

    if ob is not None: base = str(base+'&order_by='+ob)
    if so is not None: base = str(base+'&sort_order='+so)
    base += str('&api_key='+apikey+'&file_type=json')

    # Step 2: make Request to API Source
    resp = req.get(base).json()

    # Step 3: Parse search results
    head = [x.replace('\n', '') for x in open('assets/.fred_search_colnames').readlines()]
    targ = [x.replace('\n', '') for x in open('assets/.fred_search_targets').readlines()]

    body = [[] for z in head]

    for e in resp['seriess']:
        iter = 0
        for d in targ:
            if d in ['observation_start', 'observation_end', 'last_updated']:
                # _c and _u
                body[iter].append(e[d].split(' ')[0]);iter += 1
                body[iter].append(float(yahoo_calc_date(e[d].split(' ')[0])));iter += 1
            elif d == 'seasonal_adjustment':
                if e[d].find('Not'): body[iter].append(False)
                else: body[iter].append(True)
                iter += 1
            else:
                body[iter].append(e[d]);iter += 1

        # Add delay calc --> 86400 = secs/day
        body[iter].append(
            np.round((float(yahoo_calc_date(str(dt.date.today()))) -
                      float(yahoo_calc_date(str(e['last_updated'].split(' ')[0])))) / 86400, 0))

    # Step 4: Create/Save Results
    if local: return pd.DataFrame(dict(zip(head,body)))
    else: pd.DataFrame(dict(zip(head,body))).to_excel(fpath,sn,index=False)

# Wrapper to apply search (from FRED API) to params  - search text, tags, variable
def iter_fred_search(txt=None,for_iter=None,tag=None,fvar=None,fval=None,ob=None,so=None,
                     apikey=None,fdir=None,file='fred_search',sleep0=2,ret=False,add_dt=True):

    if apikey is None:
        sysout('\n(!) Error: No API key supplied (free on FRED website) - required\n')
        return None

    if for_iter is None:
        sysout('\n(!) Error: for_iter param is None (setting so not None is required) \n')

    path = ''
    if fdir is not None: path = str(fdir+'/')

    path     = ifelse(bool(add_dt),str(path+file+dt_tag()+'.xlsx'),mkstr(path,file,'.xlsx'))
    writer   = pd.ExcelWriter(path)
    # states   = get_us_states() # Old
    # states   = load_us_states() # Old
    # states = read_us_states()
    s_data   = dict()

    # states
    for i,e in enumerate(for_iter):
        sysout('> Iter_',i,': ',e,'   \t\t[ ',np.round((i+1)/len(for_iter)*100,2),'% ] ... ')

        if txt is None:
            t_sub = l_(e)
        elif isinstance(txt,(list,)):
            t_sub = [x for x in txt]
            t_sub.append(l_(e))
        else:
            t_sub = [txt,l_(e)]

        # sysout('',t_sub,'\n-----\n')

        s_data[l_(e)] = fred_search(txt=t_sub,tag=tag,fvar=fvar,fval=fval,ob=ob,so=so,apikey=apikey,
                                    fdir=fdir,file=file,
                                    sn=concat_list(['s_',i,'_',dt_tag()],sep=''),local=True)
        sysout('(*) Complete\n')
        time.sleep(int(sleep0*(1-(i/100))))

    dict2wb(writer,s_data)
    if ret: return s_data

# Clears rows that only contain False values
def tbl_clr_f(df,skip=None):
    # if skip is not None: to_save = skip
    curr_cols = list(df.columns); to_drop = []
    col_chk = [str(df.iloc[:,i]).count('T') for i in range(0,len(curr_cols))]
    for i,c in enumerate(col_chk):
        if skip is not None and i in skip: continue #isin([i],[skip])['v']: continue
        elif c <= 0: to_drop.append(i)
        else: continue

    df = df.drop(df.columns[[to_drop]], axis=1)
    return df

# Go by sheet (state), extract unique titles, tag states per dataset, determine overlap, write files
def match_titles(d,fdir=None,file={'full':'state_data_full','overlap':'state_data_overlap'},
                 store=True,retloc=False):

    # Extract States (tags = keys) + dict of only titles + prep new dict
    tags   = list(d)
    raw_t  = dict((s,d[s]['title']) for s in tags)
    # return(raw_t)
    if store: dict2wb(mkfpath(fdir,file['full'],'.xlsx'),raw_t)
    uni_t  = dict((t,list()) for t in tags)
    all_t  = []

    # Fill uni_t with 'universal' titles (no states -> for cross comparison)
    for t in tags:
        temp = raw_t[t]
        for e in temp:
            e2 = l_(e)
            # If state name not found, skip (mismatch) - add to cleaned dict + full list
            if e2.find(t) == -1: continue
            else:
                uni_t[t].append(e2.replace(t,'_state_').strip())
                all_t.append(e2.replace(t,'_state_').strip())

    # Unique Sorts them, prepare bin label structures
    all_t = list(np.unique(all_t))
    bin_t = {t:{'states':[],'t_reset':[]} for t in all_t}

    # Summary DF setup
    s_head = ['dataset', 'n_states'] + tags  # Colnames
    s_body = [[] for i in s_head]

    # Tag states and data --> data held by state in row r
    for x in all_t: # Unique Title
        sysout('> Data: ',x,'\n  [ ')
        s_body[0].append(x)
        for i,s in enumerate(list(uni_t)): # Datasets by state (check if x in dataset per state)
            sd = uni_t[s]

            if x in sd:
                bin_t[x]['states'].append(s) # State
                bin_t[x]['t_reset'].append(x.replace('_state_',s)) # Original Data
                sysout('|')
            else:
                sysout('_')

        s_valid  = bin_t[x]['states']
        n_states = len(s_valid)

        if n_states == 50: sysout(' ] - (*) 50\n\n')
        else: sysout(' ] - ',n_states, '/', len(list(raw_t)),' states\n\n')

        s_body[1].append(float(n_states))
        for i,s in enumerate(s_head[2:]):
            if s in s_valid: s_body[int(i+2)].append('T')
            else: s_body[int(i+2)].append('F')

    summdf = pd.DataFrame(dict(zip(s_head,s_body))).sort_values(['n_states'],ascending=[False])
    summdf.to_excel(mkfpath(fdir, file['overlap'], '.xlsx'),'state_overlap')

    if retloc:
        return {'full':raw_t,'unique':uni_t,'overlap':summdf}

# Filter titles by N overlaps and/or by state(s) --> Subset list of data sets to fetch data for
def filter_titles(df=None,key=None,n_range=None,s_state=None,fdir=None,retloc=False):
    ## -----
    # n_range: [n0:n1] for overlaps (if applicable, assume rationale...)
    # keys: match data set text
    # for_state, not_state: subset FOR states, excluding NOT states
    # state_iso: report individual state stats (default is False -> Not implemented yet...
    ## -----

    fpo = mkfpath(fdir,str('state_subset_ref_'+dt_tag()),'.xlsx')
    if df is None: df = pd.read_excel('state_data_overlap.xlsx')
    df.index = range(len(df.index))

    # Filter 1: Key terms (currently simple design --> to be expanded)
    if key is not None:
        sysout('> Filter: Key Text [ ',df.shape[0],' ... ')
        dst  = [list(x.split()) for x in list(df['dataset'])]
        df_f = isin(key,dst,True)
        df   = df.iloc[np.where(df['dataset']==df_f),:]

        df = tbl_clr_f(df,[0,1])
        sysout(df.shape[0],' rows ]\n')
        # return(df)

    # Filter 2: Number of Crossovers
    if n_range is not None:
        sysout('> Filter: Crossovers [ ', df.shape[0], ' ... ')
        orig = list(df['n_states'])

        r1 = int(np.where(df['n_states'] <= n_range[0])[0].min())
        r0 = int(np.where(df['n_states'] <= n_range[1])[0].min())
        df = df.iloc[r0:r1,:]

        df = tbl_clr_f(df, [0, 1])
        sysout(df.shape[0], ' rows ]\n')
        # return(df)

    # Filter 3: States - [1:50] - WIP
    if False: #s_state is not None:
        sysout('> Filter: States [ ', df.shape[0], ' ... ')
        ss = isin(s_state,list(df.columns))
        isin_c = ss['count']
        isin_v = ss['v']
        # df = df[ss]

        df = tbl_clr_f(df, [0, 1])
        sysout(df.shape[0], 'rows\n')

    return df

# Filter for data orders - numerical filters
def order_filter_n(dfd, col, fltr, floc=False, fmin=False, fmax=False, uni=True):
    dkeys = list(dfd.keys())
    vkeys = []
    vdata = []
    for i,k in enumerate(dfd.keys()):
        temp = dfd[k]
        tcol = [float(s) for s in temp[col]]
        # Check Filters: if valid, overwrite with DF else overwrite with None
        if floc:
            if uni:
                temp = ifelse(len(list(tcol == fltr)) < len(tcol), None, temp)
            else:
                temp = temp.loc[tcol == fltr,]
        elif fmin:
            if uni:
                temp = ifelse(bool(min(tcol) >= fltr), temp, None)
                # print('fmin: min(1) >= (2)',tcol,fltr,temp)
            else:
                temp = temp.loc[tcol >= fltr,]
        elif fmax:
            if uni:
                temp = ifelse(bool(max(tcol) >= fltr), None,temp)
                # print(tcol,fltr,temp)
            else:
                temp = temp.loc[tcol <= fltr,]
        else:
            temp = temp

        if isok(temp):
            vkeys.append(dkeys[i])
            vdata.append(temp)
    return dict(zip(vkeys,vdata))

# Data Fetching Helper
def fred_get_data(d, key, dn=None,stimer=5):
    ids = d['id']
    if not isok(dn): dn = [str('d_'+str(i)) for i in range(1,len(ids)+1)]

    base_url = 'https://api.stlouisfed.org/fred/series/observations?series_id='
    url_end = mkstr('&api_key=', key, '&file_type=json')

    t0 = dt.datetime.utcfromtimestamp(min(d['ob0_u'])).strftime('%Y-%m-%d')
    t1 = dt.datetime.today().strftime('%Y-%m-%d')
    so = 'asc'

    # temp = []
    for i, x in enumerate(ids):
        if i == 0:
            sysout('\n  ', len(ids), ' Unique Variables in Data Set ...')
            sysout(concat_list(
                [ifelse(i==0,
                        '\n>|',
                        ifelse(i == len(ids),
                               '| - Progress', ' ')) for i in range(0,len(ids)+1)],''))
            sysout('\n [')

        full_url = mkstr(base_url, x, '&sort_order=', so, '&observation_start=',
                         t0, '&observation_end=', t1, url_end)
        resp = req.get(full_url).json()['observations']
        if i == 0:
            out = pd.DataFrame(resp)[['date', 'value']]
            # float(yahoo_calc_date(str(e['last_updated'].split(' ')[0])))) / 86400
            out['date'] = [str(s.split(' ')[0]) for s in out['date']]
            out.columns = ['date', dn[i]]
            obu = [int(yahoo_calc_date(ud)) for ud in out['date']]
            out.insert(1, 'dateu', obu)
            # out[dn[i]] = [conv_s2n(x) for x in out[dn[i]]]
            out[dn[i]] = [conv_s2n(x) for x in out[dn[i]]]
            # return out
        else:
            df = pd.DataFrame(resp)[['date', 'value']]
            df['date'] = [str(s.split(' ')[0]) for s in df['date']]
            df.columns = ['date', dn[i]]
            df[dn[i]] = [conv_s2n(x) for x in df[dn[i]]]

            out = pd.concat([out.reset_index(drop=True), df], axis=1)

            tcn = list(out.columns)
            tcn[len(tcn) - 2] = 'date2'
            out.columns = tcn
            out = out.drop('date2', axis=1)
            sysout('|')
        time.sleep(stimer)
    sysout(']\n')
    # out.replace(r'\s+', np.nan, regex=True)
    return out

# Create dictionary-style data order to fill
def create_data_order(map,fullmap,uniform_set=True,min_pop_iso=None,min_pop_grp=None,ob0=None,ob1=None,
                      max_delay=None,apikey=None,drop_percents=True):
    if apikey is None:
        sysout('\n(!) Error: No API key supplied (free on FRED website) - required\n')
        return None

    base_url = 'https://api.stlouisfed.org/fred/series/observations?series_id='
    # full_map = wb2dict(pd.ExcelFile('fred_state_reference.xlsx'))
    full_map = wb2dict(pd.ExcelFile(mkstr(fullmap,'.xlsx')))
    url_end  = concat_list(['&api_key=',apikey,'&file_type=jason'])

    # Subset full data map into smaller one
    # map_d = map.to_dict() # If needed
    d_tags = list(map['dataset'])
    s_bool = map.iloc[:,2:]
    s_name = list(map.columns[2:])

    # Re=apply name-tags (unique data ID)
    sysout('\n> Re-creating Unique Data Tags ...')
    by_data   = [[d.replace('_state_',sn) for sn in s_name if list(s_bool[sn])[i] == 'T'] for i,d in
               enumerate(d_tags)]
    sysout(' DONE')

    # List of single row data frames (Rows matched to original Data)
    sysout('\n> Aggregating Original Data by Tags ...')
    row_data  = [[full_map[s].iloc[id_row(full_map[s], 'title', d[i]),:] for i, s in enumerate(
        s_name)] for d in by_data]
    sysout(' DONE')

    # Dictionary of lists (lists = iso rows of data by tag by state -> need to agg into DF)
    sysout('\n> Creating Dictionary of Data Sets (DF) ...')
    df_dict = dict(zip(d_tags,row_data))
    for k in df_dict.keys():
        df_dict[k] = rbind(df_dict[k])
    sysout(' DONE')

    # Apply Filters
    if isok(min_pop_iso):
        sysout('\n> Filter: Data Set Popularity ...')
        df_dict = order_filter_n(df_dict,'pop_iso',min_pop_iso,fmin=True,uni=uniform_set)
        sysout(' DONE')
    if isok(min_pop_grp):
        sysout('\n> Filter: Data Group Popularity ...')
        df_dict = order_filter_n(df_dict,'pop_group',min_pop_grp,fmin=True,uni=uniform_set)
        sysout(' DONE')
    if isok(ob0):
        sysout('\n> Filter: Start Date (T_0) ...')
        ob0_u = float(yahoo_calc_date(ob0))
        df_dict = order_filter_n(df_dict,'ob0_u',ob0_u,fmax=True,uni=uniform_set)
        sysout(' DONE')
    if isok(ob1):
        sysout('\n> Filter: End Date (T_T) ...')
        ob1_u = float(yahoo_calc_date(ob1))
        df_dict = order_filter_n(df_dict,'ob1_u',ob1_u,fmin=True,uni=uniform_set)
        sysout(' DONE')
    if isok(max_delay):
        sysout('\n> Filter: Max Delay (Days) ...')
        df_dict = order_filter_n(df_dict,'delay',max_delay,fmax=True,uni=uniform_set)
        sysout(' DONE')
    sysout('\n> [!] Returning Data Order\n')
    return df_dict

# Fill the data order
def fill_order(ord,dn=None,dir=None,apikey=None,stimer0=1,stimer1=3,add_dt=True,user_tag=None):
    fn   = ifelse(bool(add_dt),
                  str('fred_datastore_' + dt_tag()),
                  str('fred_datastore_' + user_tag))
    fp_d = mkfpath(dir, fn, e='.xlsx')
    fp_o = ifelse(bool(add_dt),
                  mkfpath(dir, f=str('fred_dataorder_' + dt_tag()), e='.xlsx'),
                  mkfpath(dir, f=str('fred_dataorder_' + user_tag), e='.xlsx'))
    fp_t = ifelse(bool(add_dt),
                  mkfpath(dir, f=str('fred_dataorder_ref_'+dt_tag()),e='.txt'),
                  mkfpath(dir, f=str('fred_dataorder_ref_' +user_tag), e='.txt'))
    dict2wb(fp_o,ord,False)

    # fp_d2 = mkfpath(dir, 'testbranch1', '.xlsx')
    # fp_d3 = mkfpath(dir, 'testbranch2', '.xlsx')
    # fp_d4 = mkfpath(dir, 'testbranch3', '.xlsx')
    sns = list(ord.keys())
    sna = [str('dataset_'+str(i)) for i in range(1,len(sns)+1)]
    tf  = open(fp_t,'w')
    tf.writelines('\n'.join([str(sns[i] + '_<(jr)>_' + sna[i]) for i in range(len(sns))]))
    tf.close()

    outf = []
    # outf.append(pd.DataFrame([sna,sns],columns=['dtag','dval']))

    # writer_d = pd.ExcelWriter(fp_d)
    # writer_d4 = pd.ExcelWriter(fp_d4)
    for i,sn in enumerate(sns):
        # writer_d2 = pd.ExcelWriter(fp_d2)
        # writer_d3 = pd.ExcelWriter(fp_d3)
        sysout('\n> Fetching Data Set  ',i+1,' of ',len(sns),': ',sn,' ...')
        df = fred_get_data(ord[sn],apikey,dn,stimer1)
        sysout(' DONE - Obs: ', len(df.index), '\n')
        outf.append(df)
        # sysout(' DONE - Obs: ',len(df.index),'\n  + Saving to: ',fp_d,' ...')
        # df.to_excel(writer_d,sheet_name=str('dataset_'+str(i+1)))
        # df.to_excel(writer_d2, sheet_name=str('dataset_' + str(i+1)))
        # df.to_excel(writer_d3, sheet_name=str('dataset_' + str(i+1)))
        # df.to_excel(writer_d4, sheet_name=str('dataset_' + str(i+1)))
        # sysout(' DONE')
        time.sleep(stimer0)
        # writer_d2.save()
        # writer_d3.save()
        # writer_d4.save()
        # writer_d2.close()
    # return [sna,outf]

    out_main = dict(zip(sna,outf))
    dict2wb(pd.ExcelWriter(fp_d),out_main)
    # writer_d.save()
    # writer_d.close()
    # writer_d4.close()
    sysout('\n-----\n> [!] Order Filled at: ',fp_d)


