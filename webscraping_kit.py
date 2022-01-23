import re, os, glob, csv, math, requests, time, sys, random, json, datetime, urllib
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from sklearn.utils import shuffle      
import pandas as pd
import numpy as np
from nordvpn_switcher import initialize_VPN,rotate_VPN,terminate_VPN
from webdriver_manager.chrome import ChromeDriverManager

def rmnl(t):
    """removes all extra space from a text"""
    return re.sub('\s+',' ',t).strip()

def write_htmlfile(response,file):
    """writes the response to a file"""
    with open(file, "w") as f:
        f.write(str(response.text))
        
def write_driverhtmlfile(response,file):
    """writes the driver response to a file"""
    with open(file, "w") as f:
        f.write(response)

def read_htmlfile(file):
    """creates a beautifulsoup object from an html file"""
    with open(file, 'r') as f:
        soup = bs(f.read(),'lxml')
    return soup

def read_htmlresponse(response):
    """creates a beautifulsoup object from a response"""
    soup = bs(response.text,'lxml')
    responsetext = rmnl(soup.text)
    return responsetext

def read_driverresponse(response):
    """creates a beautifulsoup object from a response using selenium webdriver"""
    soup = bs(response,'lxml')
    responsetext = rmnl(soup.text)
    return responsetext

def read_jsoncsv(file):
    """reads a newline-separated csv file of json/dict objects"""
    jobs = []
    with open(file,'r') as f:
        lines = f.readlines()
    for line in lines:
        jobs.append(json.loads(line))
    return jobs

def write_json_tofile(response,file):
    """writes a json/dict object to a newline-separated csv file"""
    with open(file,"a") as f:
        f.write(json.dumps(response)+'\n')
        
def get_apikeys(apikeyfile,apiname):
    """read the api keys based on the api"""
    with open(apikeyfile,'r') as f:
        apikey = json.loads(f.read())[apiname]
    return apikey
        
#--------------------------------------------------------------------------
"""There are two ways to get the list of cases available through CanLII. First, you can request an api key from them and get basic information about the case ids, citations, and titles. You can then create the links to the case texts using the case ids and then webscrape them. Second, you can manually download tables (or webscrape them) for cases by year (should sum to 36,319 as of 2021-12-28) and extract links to cases from there."""
#--------------------------------------------------------------------------
"""             1. USING THE API TO GET CANLII CASES                    """
#--------------------------------------------------------------------------
   
def build_queries(canlii_key,start,end):
    """creates requests by date range to the canlii api
    - start/end - str/int of start and end years for queries
    - the start/end dates are from ...-12-31 to -01-01"""
    daterange = list(range(int(start)-1,int(end)))
    queries = []
    for i,d in enumerate(daterange):
        monthdaystart = "-12-31"
        monthdayend = "-01-01"
        d1 = str(d)+monthdaystart
        d2 = str(d+2)+monthdayend
        dd_bef_aft = "&decisionDateBefore="+d2+"&decisionDateAfter="+d1
        query = "https://api.canlii.org/v1/caseBrowse/en/onltb/?offset=0&resultCount=10000" + dd_bef_aft + "&api_key="+canlii_key
        queries.append(query)
    return queries

def query_canlii_api(queries,apioutfile,headers):
    """sends the queries from 'build_queries(...)' to the canlii api and saves the json/dict results to a newline-separated csv file"""
    for i, query in enumerate(queries):
        canlii_resp = requests.get(query, headers=headers)
        print(canlii_resp)
        save_canlii_response(canlii_resp,apioutfile)
        time.sleep(3)
        print(i)

def save_canlii_response(canliiapiresponse,apioutfile):
    """parses the canlii api responses and writes them to a newline-separated csv file"""
    ltbdict = json.loads(str(bs(canliiapiresponse.content, "html.parser")))
    write_json_tofile(ltbdict,apioutfile)
    
def parse_caseid(cs):
    """get the canlii case ids from the api responses to use 'format_canlii_caseurls(row)'"""
    caseId = cs['caseId']
    if 'en' in caseId:
        cid = caseId['en']
    else:
        cid = caseId['fr']
    return cid
        
def parse_canlii_responses(apioutfile,casefolder):
    """parse the canlii api responses and create a dataframe"""
    caselist = []
    for job in read_jsoncsv(apioutfile):
        cases_sublist = job['cases']
        for cs in cases_sublist:
            cid = parse_caseid(cs)
            title = cs['title']
            citation = cs['citation']
            caselist.append([cid,title,citation])
    cdf = pd.DataFrame(caselist)
    cdf.columns = ['cid','fileno','citation']
    cdf['year'] = cdf['citation'].apply(lambda x: int(x.split(' ',1)[0]))
    cdf.drop_duplicates(inplace=True)
    cdf.sort_values(by='year',inplace=True)
    cdf['caseurl'] = cdf.apply(lambda x: format_canlii_caseurls(x),axis=1)
    cdf['caseoutfile'] = cdf['cid'].apply(lambda x: casefolder + 'canlii'.join(x.split(' CanLII ',1)) + '.html')
    return cdf

def format_canlii_caseurls(row):
    """after collecting the cases from the canlii api, use the case ids to find the case texts, NOT AVAILABLE VIA CANLII (CHECK!!!!!!!)"""
    burl = "https://www.canlii.org/en/on/onltb/doc/"
    curl = burl + str(row['year']) +'/' + row['cid'] + '/' + row['cid'] +'.html'
    return curl

#--------------------------------------------------------------------------
"""                 2. WEBSCRAPING CANLII DIRECTLY                    """
#--------------------------------------------------------------------------

def parse_tabledata(soups,casefolder):
    """extracts links and metadata from tables in manually downloaded yearly canlii html files"""
    tabledata = []
    for i, soup in enumerate(soups):
        s = soup[1]
        f = soup[0]
        rescount = "|".join([x.strip() for x in re.findall('\d+',s.find('p',{'id':'countMessage'}).text)])
        caserows = [i.find('div',{'class':'col'}) for i in s.findAll('div',{'class':'row row-stripped py-1 ml-0'})]
        for r in caserows:
            url = r.find('a',href=True)['href']
            fileno = r.find('a').text
            cid = r.text.split(',',1)[1].strip()
            cid = re.findall("\d{4} CanLII \d+",cid)[0]
            year = int(cid.split(' CanLII ')[0].strip())
            caseoutfile = casefolder + 'canlii'.join(cid.split(' CanLII ',1)) + '.html'
            cid = 'canlii'.join(cid.split(' CanLII ',1))
            tabledata.append([cid,year,f,rescount,url,fileno,caseoutfile])
    df = pd.DataFrame(tabledata,columns=['cid','year','tablefile','table_result_count','caseurl','fileno','caseoutfile'])
    return df

def get_tablerescount(soup):
    """get the # of results listed in each yearly table of cases"""
    tablerescount = re.findall('\d+',soup.find('p',{'id':'countMessage'}).text)
    if len(tablerescount) ==1:
        tcount = int(tablerescount[0])
    else:
        tablerescount = [int(x) for x in tablerescount]
        tcount = max(tablerescount)
    return tcount

def get_tabledata(tablelink,tablefile):
    """webscrape the yearly tables of ONLTB cases"""
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(tablelink)
    tablehtmlpage1 = driver.page_source
    tablesouppage1 = bs(tablehtmlpage1,'html.parser')
    casetotal = get_tablerescount(tablesouppage1)
    if casetotal >500:
        numclicks = list(range(1,math.ceil(casetotal/500)))
        for i in numclicks:
            showmore = driver.find_element_by_class_name('showMoreResults')
            showmore.click()
            print('\t ...sleeping for 5 seconds...')
            time.sleep(5)
            print('\t ...resuming...')
            print(i)
        tablehtml = driver.page_source
        write_driverhtmlfile(tablehtml,tablefile)
    else:
        tablehtml = driver.page_source
        write_driverhtmlfile(tablehtml,tablefile)        

#--------------------------------------------------------------------------
        
def get_origin(row,tablecids):
    """get the source of the case (using the canlii api vs webscraping, or both"""
    x = row['cid']
    sourcetag = row['source']
    newtag = []
    if sourcetag !='both':
        if x in tablecids:
            tag = 'webscrape'
            newtag.append(tag)
        else:
            tag = 'api'
    else:
        tag='both'
    return tag

def get_cases(caselinkdict,settings):
    """webscrape collected links to ONLTB cases"""
    sleeptimes = list(range(3,10))
    for i, (k,v) in enumerate(caselinkdict.items()):
        driver = webdriver.Chrome(ChromeDriverManager().install())
        driver.get(v)
        casehtml = driver.page_source
        rtext = read_driverresponse(casehtml)
        if rtext.startswith('Captcha'):
            print(v)
            rotate_VPN(settings,google_check=1)
            driver = webdriver.Chrome(ChromeDriverManager().install())
            driver.get(v)
            casehtml = driver.page_source
            write_driverhtmlfile(casehtml,k)
        else:
            write_driverhtmlfile(casehtml,k)
            continue
        sleeptime = random.sample(sleeptimes,1)[0]
        time.sleep(sleeptime)
        print(i)