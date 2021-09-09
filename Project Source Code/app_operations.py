from app_utilitiles import *


# region :: classify()
''' 
Add classifications (numeric) as levels with labels. Works with DFs (primary design) but also lists and 
dictionaries IFF the underlying components (at some depth) are DFs. This is a recursive Function for data 
stuctures: list and dictionary. This clasification deals with ALL data in the table (assumes homogenous data), so 
the default, for instance, is set to calculate the percentiles of each cell (x_ij) relative to the data set of EVERY 
cell (x_ij). Adjusts (in App) per data table change

'''
def classify(df,levels=None,tags=None,iso_col=None):
    if isinstance(df,list):
        return [classify(df[i],levels,tags,iso_col) for i in range(len(df))]
    elif isinstance(df,dict):
        dnames = list(df.keys())
        out = dict()
        for k in dnames: out[k] = classify(df[k],levels,tags,iso_col)
        return out
    elif isinstance(df,pd.DataFrame):
        df0 = df.copy()

        # Check to isolate (exclude and re-add) columns (i.e. dates/Index)
        if isok(iso_col):
            to_iso = [df.columns[i] for i in iso_col]
            df2 = df[to_iso]
            df = df.drop(to_iso,axis=1,inplace=False)
        else: df2 = None

        # Check if levels are provided
        if isok(levels) and isinstance(levels,list): levels = levels
        else:
            d_agg = aggregate(df, by_row=False)
            levels  = list(np.percentile(d_agg,list(range(1,101))))

        # Check if tags are provided (only use if levels was also a valid input)
        if isok(tags) and isok(levels): tags = tags
        else: tags = [i for i in range(1,(len(levels)+1))]

        # Apply Classifications
        temp_all = list()
        for c in list(df.columns):
            d = list(df[c])
            temp = []
            for e in d:
                for i,lev in enumerate(levels):
                    if e <= lev:
                        temp.append(tags[i])
                        break
                    if e > lev and i==len(levels):
                        temp.append(tags[i])
                        break
            # print('loop done ----------')
            temp_all.append(temp)

        df2 = cbind(df2,dict(zip(list(df.columns),temp_all)))
        return dict(orig=df0,df_class=df2,levels=levels,tags=tags)
    else:
        sysout('\n> **[!] Error: Bad Input Type - not list/dict/df')
        return None
# endregion


def durl_irs_1(y):
    if isinstance(y,list): return [durl_irs_1(x) for x in y]
    else: return mkstr('https://www.irs.gov/pub/irs-soi/',y,'in54cmcsv.csv')

def get_irs_data1(dir=None,local=False):
    fp = ifelse(isok(dir),
                mkstr(dir,'/data_irs1.xlsx'),
                'data_irs1.xlsx')
    df = [pd.read_csv(durl_irs_1(y)) for y in range(12,17)]
    dd = mkdict([mkstr('20',y) for y in range(12,17)],df)
    dict2wb(fp,dd)
    if local: return dd