"""
Created on Sat Sep 29 17:16:20 2018

@author: Wyatt
"""

from yahoofinance import *

# Fetch Ticker ETF Exposure by % in those ETFs
def get_etf_exposure(t):
	# Raw HTML parsing from url
	url = 'https://etfdailynews.com/stock/%s/' % (t.upper())
	src = req.get(url).content
	raw = BeautifulSoup(src, features='html.parser')
	tbl = raw.find("table", {"id": "etfs-that-own"})

	# return(tbl)
	# Headers (Column names)
	head0 = tbl.findAll('th')
	head1 = [h.string for h in head0]
	# print('Headers: ',head1)
	# Values (Column values)
	rows = tbl.findAll('tr')
	etbl = [[] for t in range(0, 3)]
	for row in rows[1:]:
		elems = row.findAll('td')
		ehold = [e.string for e in elems]
		etbl[0].append(ehold[0])
		etbl[1].append(ehold[1])
		etbl[2].append(ehold[2])

	# edict = dict(zip(head1,etbl))
	tdict = dict(zip(['head', 'body', 'iter'], [head1, etbl, [i for i in range(0, len(etbl[0]))]]))
	# return(etbl)
	return (tdict)

# Fetch Company Competitors
def get_competitors(t):
	url = 'https://www.nasdaq.com/symbol/%s/competitors' % (t.upper())
	src = req.get(url).content
	raw = BeautifulSoup(src, features='html.parser')
	tbl = raw.findAll('table')[2]

	# Head (Column names)
	head = [re.sub('[\t\n\r\xa0▲\xa0▼" "":"]+', ' ', e.get_text().strip())
			for e in tbl.findAll('th')]
	# print(head)
	head = head[0].split(' ') + head[1:4] + head[4].split('/') + \
		   head[5].split(' / ') + head[6:]
	head = head[0:7] + ["Todays's" + head[7], head[8], "52 Weeks " + head[9], head[10].strip(' '),head[11]]
	# print(head)
	# print(len(head))

	# Values (Columns values)
	vals = [[] for ti in range(0, len(head))]
	for itr, etr in enumerate(tbl.tbody.findAll('tr')):
		# print('>> (tr): '+str(itr))
		etd = etr.findAll('td')
		loc = 0
		for itd, etd in enumerate(etd):
			# print('>> (td): '+str(itd))
			ess = etd.stripped_strings
			for istr, v in enumerate(ess):
				# print('>> String:'+str(loc)+':'+v)
				if loc == 0:
					vals[loc].append(v)
				else:
					vals[loc].append(re.sub('[\t\n\r\xa0▲\xa0▼" "":"]+', '', v))
				loc += 1

	tdict = dict(zip(['head','body','iter'],[head,vals,[i for i in range(0,len(vals[0]))]]))
	return(tdict)

# Fetch Current Yield Curve Data + Create Plot Figure Output
def get_yc():
	url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield'
	src = req.get(url).content
	raw = BeautifulSoup(src, 'html')
	tbl = raw.findAll('table')[0]
	# Head (Column Names)
	head = [elem.string.strip() for elem in tbl.findAll('th')]
	head.remove('2 mo')
	# Contents
	vals = [[] for i in range(0, len(head))]
	# print(len(vals))
	for row in tbl.findAll('tr')[2:]:
		loc = 0
		for ei,elem in enumerate(row.findAll('td')):
			if ei == 2 or elem.string is None: # 2 month (2 mo) note yield
				continue
			vals[loc].append(elem.string)
			loc += 1
	# Yield Curve Plot (Most recent date)
	curr_date = vals[0][-1]
	curr_yields = []
	for y in vals[1:]:
		curr_yields.append(float(y[-1]))

	# print(head[1:])
	# print(len(head[1:]))
	# print(curr_yields)
	# print(len(curr_yields))

	ts = pd.Series(curr_yields,index=head[1:])
	fig = plt.figure(); ts.plot()
	plt.title('US Treasury Yield Curve as of: '+curr_date)
	img = io.BytesIO(); plt.savefig(img,format='png'); img.seek(0)
	plt_img = base64.b64encode(img.getvalue()).decode()

	gc.collect()

	# Fill output disctionary and return it
	tdict = dict(zip(['head','body','plot','iter'],[head,vals,plt_img,[i for i in range(0,
																					len(vals))]]))
	return(tdict)

# Fetch top country market rates data
def get_centralrates():
	url1 = 'https://www.investing.com/central-banks/'
	src = req.get(url1, headers={'User-agent':'Mozilla/5.0'}).content
	raw  = BeautifulSoup(src,'html')
	tbl = raw.findAll('table')[0]
	# Head (Column Names)
	head = []
	for th in tbl.thead.findAll('th'):
		ss = th.stripped_strings
		for elem in ss:
			head.append(elem)
	head.insert(1, 'Symbol')
	# print(head)
	# Values (Column Content)
	vals = [[] for i in range(0, len(head))]
	for tr in tbl.findAll('tr')[1:]:
		loc = 0
		for elem in tr.stripped_strings:
			vals[loc].append(elem)
			loc += 1
	# print(vals)
	tdict = dict(zip(['head', 'body', 'iter'], [head, vals, [i for i in range(0, len(vals[0]))]]))
	return (tdict)








