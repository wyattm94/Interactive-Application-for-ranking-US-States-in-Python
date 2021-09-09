# -*- coding: utf-8 -*-
"""
__________________________________________________________________________________________________

    > Script ID     : fred_run.py
    > Project ID    : assignment4_mapviz
    > Author(s)     : Wyatt Marciniak [wyattm94@gmail.com]
    > Maintainer(s) : Wyatt Marciniak
    > License(s)    : Copyright (C) Wyatt Marciniak - All Rights Reserved
    > Description   : Data Sourcing Script utilizing the FRED Database API and custom wrappers/parser functionalities
        >> Runable locally for best results --> Data extraction/storage is pre-set for the FRED App (v2)
        >> The data sets are already saved  in the root directory for the apps to use
        >> [WARNING] If you alter the parameters and run without changing write-paths, you will overwrite data
        >> [NOTICE] If you do that... see fred_assets/backups directory for copies of the original data sets
__________________________________________________________________________________________________
<<< Source Code Begins >>>
"""

# region :: Dependencies
from fred_api import *
fred_api_key = open('assets/.fred_api_key','r').readline()
# endregion <Dependencies>

# region ::----- RUN PHASE 1: Collect ALL dataset info per state within filter parameters

# Fetch to Memory (1) then read and extract state_names for iteration (2)
fetch_us_states()
states_text = list(xb2df('fred_warehouse/us_states.xlsx')['us_states']['state_name'])
# endregion

# region ::----- RUN PHASE 2: Compare datasets across ALL states --> create overlap summary
search_results1 = iter_fred_search(txt=['US'], for_iter=states_text,tag=['State','Monthly'],
                                  fvar='seasonal_adjustment', fval='Not Seasonally Adjusted',
                                  ob='popularity',so='desc', ret=True,apikey=fred_api_key,
                                  fdir=None, file='state_search_results_m',add_dt=False)

search_results2 = iter_fred_search(txt=['US'], for_iter=states_text,tag=['State','Annual'],
                                  fvar='seasonal_adjustment', fval='Not Seasonally Adjusted',
                                  ob='popularity',so='desc', ret=True,apikey=fred_api_key,
                                  fdir=None, file='state_search_results_a',add_dt=False)

search_ref_m = match_titles(search_results1,file=dict(full='state_data_full_m',
                                                     overlap='state_data_overlap_m'),retloc=True)
search_ref_a = match_titles(search_results2,file=dict(full='state_data_full_a',
                                                     overlap='state_data_overlap_a'),retloc=True)
# endregion

# region ::----- RUN PHASE 3: Filter data and re-use the full search data for data extraction
filtered_overlap_summ_m = filter_titles(search_ref_m['overlap'],n_range=[49,50])
filtered_overlap_summ_a = filter_titles(search_ref_a['overlap'],n_range=[49,50])
# endregion

# region ::------ RUN PHASE 4: Data Extraction, filtering and cleaning
data_order_m = create_data_order(filtered_overlap_summ_m,'state_search_results_m',
                               ob0='2005-01-01',ob1='2017-01-01',max_delay=35,uniform_set=True,
                                 apikey=fred_api_key)
data_order_a = create_data_order(filtered_overlap_summ_a,'state_search_results_a',
                                 ob0='2005-01-01', ob1='2017-01-01', max_delay=(2*365), uniform_set=True,
                                 apikey=fred_api_key)


fill_order(data_order_m,states_text,apikey=fred_api_key,stimer0=2,stimer1=1,
           add_dt=False,user_tag='m')

fill_order(data_order_a,states_text,apikey=fred_api_key,stimer0=2,stimer1=1,
           add_dt=False,user_tag='a')
# endregion

# region ::----- RUN PHASE 5: Adjust Data for integration (Anuual and Monthly)
## Read
da = wb2dict('fred_datastore_a.xlsx')
dm = wb2dict('fred_datastore_m.xlsx')

## Copy
da2 = da.copy()
dm2 = dm.copy()

## Set Parameter Limits - rdrop() drops rows by selection or range-filter
d0 = int(yahoo_calc_date('2005-01-01'))
d1 = int(yahoo_calc_date('2017-01-01'))

dm2_sub = rdrop(dm2,by=dict(target='dateu',min=d0,max=d1),keep=True)
da2_sub = rdrop(da2,by=dict(target='dateu',min=d0,max=d1),keep=True)

## Strip date and dateu columns (as copies) to add to annual-2-monthly data
c2add = [mkdict(list(x.columns),
                list(x[list(x.columns)[i]] for i in range(len(x.columns))))
         for x in [dm2_sub[k][['date','dateu']] for k in list(dm2_sub.keys())]][0]
c2add = [c2add for i in range(len(da2_sub))]

## Mutate Data Set - Annual to Monthly and remove duplicates (save to root)
da2_a2m = annual2monthly(da2_sub,2,0,c2add=c2add)

da2_iso = rmv_state_dups(dm2_sub,da2_a2m,'fred_dataorder_ref_m','fred_dataorder_ref_a')
dict2wb('fred_datastore_iso_a.xlsx',da2_iso)

## Rename the keys for dataset 2 (Annual), create a new reference
da2_iso = usekeys(da2_iso,list(da2_iso.keys()),)
new_tags = [mkstr('dataset_',len(dm2_sub.keys())+i) for i in range(1,len(da2_iso.keys())+1)]

da2_final = newkeys(da2_iso,new_tags) # Keys renamed

# endregion

# region ::----- RUN PHASE 6: Integrate All data into final Source Set - [RAW]

fn = 'fred_warehouse/fred_app_sourcedata'
all_tags = list(dm2_sub.keys()) + list(da2_final.keys())
all_data = [v for k,v in dm2_sub.items()] + [v for k,v in da2_final.items()]

conm = open('fred_dataorder_ref_m.txt','r')
cona = open('fred_dataorder_ref_a.txt','r')
ref0_m = [x.strip('\n').split('_<(jr)>_')[0] for x in conm.readlines()]; conm.close()
ref0_a = [x.strip('\n').split('_<(jr)>_')[0] for x in cona.readlines()]; cona.close()

all_refn = ref0_m + ref0_a
all_refn_sub = []
for x in all_refn:
    if x not in all_refn_sub: all_refn_sub.append(x)

## Get Units for all data sets
ord_m = wb2dict('fred_dataorder_m.xlsx')
ord_a = wb2dict('fred_dataorder_a.xlsx')
iso_a = wb2dict('fred_datastore_iso_a.xlsx')

units_m = [x['units'][0] for k,x in ord_m.items()]
iso_keys = list(iso_a.keys())
units_a = [ord_a[k]['units'][0] for k in iso_keys]

unit_all = units_m + units_a

## Aggregate - to be used locally and/or for testing if needed
master = mkdict(all_tags,all_data)
rmaster = pd.DataFrame(mkdict(['id','tag','units'],
                    [all_tags,all_refn_sub,unit_all]))

## Write to static (root) memory in [DIR: fred_warehouse/]
dict2wb(mkstr(fn,'.xlsx'),master)
rmaster.to_excel(mkstr(fn,'_ref.xlsx'),'fred_app_sourcedata_ref',index=False)

# endregion

# To pull (Current)
# master = xb2py('fred_warehouse/fred_app_sourcedata.xlsx',12)
# rmaster = xb2py('fred_warehouse/fred_app_sourcedata_ref.xlsx')

# region :: Additonal Pre-processing
from app_utilitiles import *
weights = calcweights(master,0.10,0.90,None)
weights.to_excel('fred_warehouse/fred_app_weights0.xlsx')
# endregion



