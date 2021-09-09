# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 00:16:17 2018

@author: Wyatt Marciniak
"""

import time
import os
import sys
import re
import gc
import io
import base64
import openpyxl
import datetime as dt
from datetime import datetime as dtt
import requests as req
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from helper import *
from stockinfo import *


# Calculate UNIX dates to parse Yahoo fetch URL
def yahoo_calc_date(d):
	if isinstance(d,(str,)):
		convd   = [int(x.split(' ')[0]) for x in d.split('-')]
		if int(convd[0]) < 1970: return 0 # Special Case
		current = dtt(convd[0],convd[1],convd[2])
		d_unix  = [str(x) for x in str(time.mktime(current.timetuple())).split('.')][0]
		return d_unix
	else:
		# print('else')
		d_unix  = [str(x) for x in str(time.mktime(d.timetuple())).split('.')]
		# d_unix = d
		return d_unix[0]

# Format data response dates (from timestamp)
def yahoo_format_date(d):
	dates = []
	for x in d:
		dates.append(dtt.fromtimestamp(x).strftime('%y-%m-%d %H:%M:%S'))
	return dates

# Get Market Data (OHLC + Volume + Adjusted Close, Dividends, Splits)
def yahoo_get_data(t,d='all',f='d',start='1970-1-1',end=dtt.today()):
	base = 'https://query1.finance.yahoo.com/v8/finance/chart/%s' % (t)

	# Error Handling on data/freq parameters
	if not d in ['all','price','div','split']:
		print('Bad data selection...'); return None
	if not f in ['d','w','m']:
		print('Bad freq...'); return None

	# Calculate url-parsed params
	def f_switch(f):
		switch = {'d':'1d','w':'1wk','m':'1mo'}
		return switch.get(f)
	freq = f_switch(f)
	t0   = yahoo_calc_date(start)
	t1   = yahoo_calc_date(end)

	# Parse fetching URL and GET data response (return as .json)
	url = ("""{}?&lang=en-US&region=US&period1={}&period2={}&interval={}
		&events=div%7Csplit&corsDomain=finance.yahoo.com""")
	url = url.format(base,t0,t1,freq)

	res = req.get(url).json() # Returns JSON (Call from Server)
	return(res)

# Extract OHLC + Volume + Adjusted Close data
def extract_price_data(r):
	dateu = r['chart']['result'][0]['timestamp']
	datec = yahoo_format_date(dateu)

	raw = r['chart']['result'][0]['indicators']
	adj = raw['adjclose'][0]['adjclose']
	prc = raw['quote'][0]

	keys = ['dateu','datec','open','high','low','close','volume','adjusted']
	vals = [dateu,datec,np.array(prc['open']),np.array(prc['high']),
		 np.array(prc['low']),np.array(prc['close']),np.array(prc['volume']),
		 np.array(adj)]

	return dict(zip(keys,vals))

# Extract Dividend Data (with dates)
def extract_div_data(r):
	try:
		raw = r['chart']['result'][0]['events']['dividends'] #['splits']
	except:
		return False

	amount = list()
	dateu = list()
	for x in sorted(raw):
		amount.append(raw[x]['amount'])
		dateu.append(raw[x]['date'])
	datec = yahoo_format_date(dateu)
	return {'dateu':np.array(dateu),'datec':datec,'amount':np.array(amount)}

# **Main wrapper for stock pulling
def yget_stock(t,f='d',start='1970-1-1',end=dtt.today()):
	# Adjust ticker to upper case + start date for pandas plotting
	t.upper()
	d0 = start.split('-'); d1 = (str(end).split(' '))[0].split('-')
	d0 = d0[1]+'/'+d0[2]+'/'+d0[0]; d1 = d1[1]+'/'+d1[2]+'/'+d1[0]
	# Fetch data + return data set (with input details)
	fetch = yahoo_get_data(t=t,f=f,start=start,end=end)
	price = extract_price_data(fetch)
	divs  = extract_div_data(fetch)
	return {'price':price,'dividend':divs,'ticker':t,'start':d0,'end':d1,'freq':f}

# **Main wrapper for FX data pulling
def yget_fx(t, f='d', start='1970-1-1', end=dtt.today()):
	url = 'https://query1.finance.yahoo.com/v8/finance/chart/'
	# Error Handling on data/freq parameters
	if not f in ['d', 'w', 'm']:
		print('Bad freq...')
		return None
	# Calculate url-parsed params
	def f_switch(f):
		switch = {'d': '1d', 'w': '1wk', 'm': '1mo'}
		return switch.get(f)
	freq = f_switch(f)
	t0 = yahoo_calc_date(start)
	t1 = yahoo_calc_date(end)
	# Adjust ticker to upper case + start date for pandas plotting
	t.upper()
	d0 = start.split('-'); d1 = (str(end).split(' '))[0].split('-')
	d0 = d0[1] + '/' + d0[2] + '/' + d0[0]; d1 = d1[1] + '/' + d1[2] + '/' + d1[0]
	# Parse FX 'ticker' input
	if '/' in t:
		split = t.split('/')
		if split[0] == 'USD':
			url = url + split[1] + '=x'
		else:
			url = url + split[0] + split[1] + '=x'
	else:
		url = url + t

	# Parse fetching URL and GET data response (return as .json)
	end = ("""{}?&lang=en-US&region=US&period1={}&period2={}&interval={}
		&events=div%7Csplit&corsDomain=finance.yahoo.com""")
	url = end.format(url, t0, t1, freq)

	res		= req.get(url).json()
	price	= extract_price_data(res)
	return {'price': price, 'dividend': None, 'ticker': t,'start':d0,'end':d1,'freq':f}

# Rev/EPS Estimates (Tables)
def yf_estimates(t):
	# [1] Parse URL - Request - Response
	url = 'https://finance.yahoo.com/quote/%(t)s/analysis?p=%(t)s?sortname=marketcapitalizationinmillions&sorttype=1' % {'t':t.upper()}
	src = req.get(url).content
	raw = BeautifulSoup(src,'html')
	tbl = raw.findAll('table')[0:2]
	# [2] Extract Data from HTML page response
	result = list()
	for t in tbl:
		head0 = t.findAll('th')
		head1 = [h.span.string for h in head0]
		# print('Headers: ', head1)
		rows = t.findAll('tr')
		etbl = [[] for ti in range(0, len(head1))]
		for row in rows[1:]:
			elems = row.findAll('td')
			ehold = [e.string for e in elems]
			# print('Row: ', ehold)
			for i in range(0, len(ehold)):
				etbl[i].append(ehold[i])
		result.append({'head':head1,'body':etbl})
	return(result)

# Basic plotting wrapper for this module's data fetching functions
def yf_plot(sd,what='adjusted',how='mkt'): #mkt,period,growth
	def h_switch(h):
		switch = {'mkt':what+' market data',
			'period':what+' periodic returns (%)',
			'growth':what+' relative growth (% from time 0)'}
		return switch.get(h)
	def f_switch(f):
		switch = {'d':'daily','w':'weekly','m':'monthly'}
		return switch.get(f)
	def y_alter(s):
		if how == 'mkt':
			return s
		elif how == 'period':
			temp = []
			for i in range(1,len(s),1):
				temp.append((s[i]-s[(i-1)])/s[(i-1)])
			return temp
		else:
			temp = []
			for i in range(0,len(s),1):
				if i == 0:
					temp.append(0)
				else:
					temp.append((s[i]-s[0])/s[0])
			return temp

	tick = sd['ticker']
	freq = f_switch(sd['freq'])
	yraw = sd['price'][what]
	xraw = pd.date_range(start=sd['start'],end=sd['end'],periods=len(yraw)) #,periods=len(yraw));
	# xadj
	#  = 0
	yadj = y_alter(yraw)
	if len(yadj) < len(yraw):
		xadj = xraw[1:]
	else:
		xadj = xraw
	hadj = h_switch(how)

	ts = pd.Series(yadj,index=xadj)
	fig = plt.figure()

	ts.plot()

	plt.title(str(tick+' '+freq+' time series: '+hadj))
	# plt.close(fig)
	return fig

# Send to (1), Fetch from (0) and Delete from (-1) cache (global dictionary
def do_cache(x,xc,io=1,local_cf=None):

	if local_cf is not None: cflocal = local_cf
	else: globals()['cflocal'] = dict()

	cache_flag = get_tf(pcf(cflocal,'opts','use_cache'))
	if cache_flag:
		global cache
		loc = str(pcf(cflocal, 'path', 'warehouse') + xc + '/' + x + '.xlsx')
		if io == 1:
			sysout('\n> Caching...')
			temp = wb2dict(loc)
			cache[x] = temp
			sysout(' [*] Complete\n')
		elif io == 0:
			sysout('\n> Fetched\n')
			return cache[x]
		elif io == -1:
			sysout('\n> Deleting from Cache...')
			if x in list(cache):
				del cache[str(xc + '_' + x)]
				sysout(' [*] Complete\n')
			else:
				sysout(' [!] Data Not in Cache\n')
		else:
			sysout('\n(!) Cache of ', x, ' failed - Bad IO command\n')
	else:
		sysout('\n(!) Caching is not set to run - Check config file\n')

# do_cache('aapl','equity',-1)

def qso(q,*args):
	if q: sysout(*args)

# Get Asset Class of ticker input (Currently: Equity/ETF/FX)
def det_asset_class(t):
	if t.find('/') != -1:
		return 'fx'
	else:
		url = str('https://etfdailynews.com/etf/'+u_(t)+'/')
		tag = l_(wbget(url,'select',None,'.sixteen')[0].text)
		# raw = BeautifulSoup(req.get(url).content, features='html.parser')
		# tag = l_(raw.select('.sixteen')[0].text)
		if tag == 'stock': return 'equity'
		else: return tag
# a = det_asset_class('aapl')

def fetch_assets(tickers=[],by='guest',start='1970-1-1',end=dtt.today(),q=False,
				 local_cf=None):
	# [*] Setup files/dir
	t_elap0 = time.time(); dir_flag=True; upd_flag, skip_flag = False, False

	sysout('\n> Determining Asset Classes...')
	asset_req = {'equity':[],'rtf':[],'fx':[]}
	for t in asset_req:
		ac = det_asset_class(t)
		asset_req[ac].append(t)
	sysout(' Complete\n-----\n')

	if local_cf is not None: cflocal = local_cf
	else: globals()['cflocal'] = dict()

	# [*] Loop through asset types (x keys)
	# qso(q,'\nto ac loop 1\n')
	for aci,ac in enumerate(list(asset_req)):
		t = asset_req[ac] # Current AC tickers
		sleep_base = max(1, np.round(len(t) / 10)); sleep_curr = sleep_base; time_last = 0
		f_dir = str(pcf(cflocal,'path','warehouse')+l_(ac)+'/')

		# sysout('-----\n(!) Current Base Sleep Interrupt set to ', sleep_base, '\n-----\n')
		sysout('> Asset Class: ', ac, ' - ',(aci+1),'/',len(list(asset_req)),'\n-----\n')

		# [*] Loop through tickers in asset class

		if not isinstance(t,(list,)):
			temp_t = []; temp_t.append(t); t = temp_t

		for i, e in enumerate(t):
			sysout('  + ', e.upper(), ' (', str(i + 1), '/', len(t), ') ... ')
			cfn = str(f_dir+e.lower()+'.xlsx')

			writer = pd.ExcelWriter(cfn) # Writer - warehouse
			logcon = pd.ExcelWriter(pcf(cflocal, 'path', 'logfile'))  # writer - logfile
			logdf  = wb2dict(pcf(cflocal, 'path', 'logfile'))  # Copied log file (1 x 1)
			aclog  = logdf[ac]  # Current directory entries by AC filter
			ft0 = time.time()

			sysout('\n  + Directory Exists')

			t_in_dir = e.lower() in str(aclog['ticker'])
			sysout('  | ', u_(e), ' Found - ', t_in_dir, '\n')

			if t_in_dir:
				txr = float(np.where(aclog['ticker'] == e.lower())[0])
				txd = float(aclog.loc[txr, 'updated_u'])

				# Check for needed updates
				update_threshold = float(pcf(cflocal, 'ctrl',
										'update_time')) * 86400  # n days x seconds in 1 day
				if (float(yahoo_calc_date(dtt.today())) - txd) >= update_threshold:
					upd_flag = True

				if upd_flag:
					# new_t0 = str(yahoo_format_date([txd]))
					new_t0 = str(aclog.loc[txr, 'updated_c'])
					new_t1 = str(dtt.today())
				else:
					skip_flag = True  # NO UPDATE
					sysout('\n  + Update Possible: ', upd_flag)

			if skip_flag:  # Cache, increase iterators (i,e,etc...) by 1
				# if str(ac+'_'+e.lower()) not in list(cache):
				# 	sysout(' - Caching...\n-----\n')
				# 	# do_cache(e.lower(),ac,1,cflocal)
				continue # Next Iteration

			if upd_flag:  # Get copy of the old data, append to it then store/cache
				sysout('\n  + Copying Old Data\n-----\n')
				df0 = wb2dict(cfn)  # wb --> dict

			# [***] Data Scrape Start
			sysout('\t+ Price & Dividend Data:')
			for freq in ['d','w','m']:
				sysout('\n\t  + frequency: ',freq)
				ft0 = time.time()

				# [*] Update flag affects date range --> append new data to bottom of old DF
				# [*] Update flags affects write to excel, or append and write
				if upd_flag:
					sysout(' | Fetch Missing')
					temp = yget_stock(e,f=freq,start=new_t0,end=new_t1)
					local2 = True
				else:
					temp = yget_stock(e,f=freq,start=start,end=end)
					local2 = False

				for d in list(temp.keys()):
					if type(temp[d]) is dict:
						sn = str(e.lower() + '_' + d + '_' + freq)
						temp_dict = {'head':list(temp[d].keys()),'body':temp[d]}

						# Catch for local (when applicable)
						catch_upd = ws_raw2df(temp_dict,
											  list([0]+list(range(2,len(temp_dict['head'])))),
											  sort='dateu', xc_writer=writer, fp=None,
											  sn=sn,
											  local=local2)

						# [*] If updating, append DFs, re-write warehouse
						if upd_flag:
							sysout(' | Append Local')
							df0[sn] = df0[sn].append(catch_upd)
							dict2wb(writer,df0)


						time.sleep(2)
				sysout('  [*] Completed in ', np.round(time.time() - ft0, 3), ' seconds')
			sysout('\n-----\n')


			# Stats
			fts = time.time() - ft0
			# Equity stats
			if not upd_flag and ac not in ['etf','fx']: # Run only for new equity extractions

				sysout('\n> Rev/EPS Est...')
				snl = ['eps_est','rev_est']
				ft0 = time.time()
				temp = yf_estimates(e)
				for j,dtemp in enumerate(temp):
					ws_raw2df(dtemp,numcols=list(range(1,len(dtemp['body']))),xc_writer=writer,fp=None,
							  sn=str(e.lower() +'_'+ snl[j]))
				sysout('\t [*] Completed in ', np.round(time.time() - ft0, 3), ' seconds')


				sysout('\n> Mkt/Fin Metrics...')
				ft0 = time.time()
				get_finmetrics(e,xcon=writer)
				sysout(' [*] Completed in ', np.round(time.time() - ft0, 3), ' seconds')


				sysout('\n> ETF Exp (%)...')
				ft0 = time.time()
				ws_raw2df(get_etf_exposure(e),[2], xc_writer=writer, fp=None,
						  sn=str(e.lower() + '_etf_exp'))
				sysout('\t [*] Completed in ',np.round(time.time() - ft0, 3),' seconds')


				sysout('\n> Comp List...')
				ft0 = time.time()
				ws_raw2df(get_competitors(e),list(range(3,12)), xc_writer=writer, fp=None,
						  sn=str(e.lower() + '_comps'))
				sysout('\t\t [*] Completed in ', np.round(time.time() - ft0, 3),
					   ' seconds\n-----\n')

			# {*} Saving/Cachine/Logging
			writer.save()  # Flush sheets into workbook
			writer.close() # Close Connection (Memory Leaks/Perf. Issues)
			# do_cache(e.lower(), ac, 1,cflocal)

			# [*] Update (or add) log entry
			log_row = [e.lower(),float(yahoo_calc_date(dtt.today())),str(dtt.today()),
								str(by),str(cfn)]


			# if not dir_flag: logdf.loc[i] = log_row
			if upd_flag:
				# aclog.loc[float(np.where(aclog['ticker'] == e.lower())[0])] = log_row
				aclog.loc[aclog['ticker'] == e.lower()] = [log_row]
			else:
				tempdf = pd.DataFrame(log_row).transpose()
				tempdf.columns = list(aclog)
				aclog  = aclog.append(tempdf)
				logdf[ac] = aclog
				# return logdf
				dict2wb(logcon,logdf)

			# logcon.save()
			# logcon.close()

			# [2b] Determine sleep time adjustments (if needed)
			if i == 0: time_last = fts
			elif (fts-time_last)/time_last >= 0.25:
				sleep_curr += 1
				time_last = fts
				# sysout('(!) Speed Adjustment - Sleep Time Increased to ',sleep_curr,'\n')
			elif (fts-time_last)/time_last < 0:
				sleep_curr = max((sleep_curr - 1),sleep_base)
				time_last = fts
				# sysout('(!) Speed Adjustment - Sleep Time Decreased to ', sleep_curr, '\n')
			else:
				sleep_curr += 0
				time_last = fts
			time.sleep(sleep_curr)

			# # Re-writing/storing updated/loaded data & logs
			# if not dir_flag:
			# 	logdf.to_excel(logcon,sheet_name=ac,index=False)
			# else:

	# End (Full elapsed Time)
	sysout('-----\nTotal Operation Time: ',np.round(time.time()-t_elap0,3),' seconds\n-----\n\n')

# Get Financial Key metrics and Q/Y freq report summaries + growth (will later fetch ALL statements)
def get_finmetrics(e,xcon):
	urls = [str('https://stockrow.com/api/companies/'+l_(e)+'.json?ticker='+l_(e)),
			str('https://www.finviz.com/quote.ashx?t=' + l_(e))]
	for i,u in enumerate(urls):
		raw = BeautifulSoup(req.get(u).content, 'html')
		if i == 0:
			temp = pd.read_json(u).transpose()
			temp2 = pd.DataFrame({'metrics': list(temp.index),
								  'values': list(temp[temp.columns[1]])}).drop([9, 10])
		else:
			temp3 = temp2.append(pd.DataFrame(
				{'metrics':[x.get_text() for x in raw.select('.snapshot-td2-cp')],
				'values':conv_s2n([x.get_text() for x in raw.select('.snapshot-td2')],['AMC',
																					   'AMO',
																					   'BMO',
																					   'BMC'])}))
	temp3.to_excel(xcon,str(l_(e)+'_keymetrics')) # Write to workbook

	# [2] Metrics and Growth, Q and Y frequencies
	base = 'https://stockrow.com/api/companies' \
		   '/{}/financials.xlsx?dimension=MR{}&section={}&sort=asc'
	for d in ['Metrics','Growth']:
		for f in ['Q','Y']:
			z = pd.read_excel(base.format(u_(e),f,d)).to_excel(xcon,str(l_(e)+'_'+l_(d)+'_'+l_(f)))





### END-----
# a = pd.DataFrame(['a','b','c']).transpose()
# a.columns = ['a','b','c']
# list(a)

# test_assets = {'equity':['aapl','hpe'],'etf':['spy','xlb']}
# test_assets = {'equity':['aapl'],'etf':['spy']}
# load_config()
# a = fetch_assets(test_assets)
