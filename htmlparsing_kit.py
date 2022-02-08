import re, os, glob, csv, math, requests, time, sys, random, json, datetime, urllib, nltk
from bs4 import BeautifulSoup as bs
from sklearn.utils import shuffle     
import multiprocessing as mp
from selenium import webdriver
from tqdm import tqdm
import numpy as np
import pandas as pd
from string import digits
from stop_words import get_stop_words
from nordvpn_switcher import initialize_VPN, rotate_VPN, terminate_VPN
from nltk.corpus import stopwords
from nltk.util import everygrams
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

from webscraping_kit import write_htmlfile, write_json_tofile, write_driverhtmlfile, rmnl
from webscraping_kit import read_jsoncsv, read_htmlfile, read_htmlresponse, read_driverresponse


"""case webscraping and parsing functions"""

def divide_chunks(l, n):
    """splits a list into chunks of size n """
    for i in range(0, len(l), n): 
        yield l[i:i + n]
        
def parse_soups(soups,metadataoutfile):
    """extracts metadata from case htmls"""
    for gs in soups:
        gs = gs.split('|',1)
        file = gs[0]
        cid = file.split('/')[-1].split('.html',1)[0].strip()
        s = bs(gs[1],'lxml')
        if not gs[1].startswith('ERROR_'):
            try:
                text = rmnl(s.text).split("var urlToCall",1)[0].strip().split("Home › Ontario › Landlord and Tenant Board ›",1)[1].strip().split('SUMMARY OF CALCULATIONS',1)[0].strip().split("If you have any questions about this",1)[0].strip().split("The tenant has until",1)[0].strip().split("The Tenant has until",1)[0].strip().split('This order contains all',1)[0].strip().split('Reasons for this order are attached',1)[0].strip().split('I f you have any questions about this',1)[0].strip().split('If you have any question s about',1)[0].strip().split('The tenants have until',1)[0].strip().split('In accordance with',1)[0].strip()
                datesplitter = r"Date Issued|Date Issue|Date Amended|Date Order Issued|Date Reasons were Issued|Date issued|_________________|Original Order|Date Ordered Issued"
                memberloc = get_memberlocdiv(text)
                member = get_member(memberloc)
                filenos = get_filenos(text)
                loc = get_loc(memberloc)
                t = text.split('Expanded Collapsed',1)[1].strip()
                sections = t.split('File Number',1)[0].strip().split('Order under',1)[-1].strip()
                date = get_date(rmnl(s.text))
                casedict = {"cid":cid,"date":date,"member":member,"memberloc":memberloc,"filenos":filenos,"loc":loc, "sections":sections,"text":t,"otext":rmnl(s.text),"error":"no_error"}
                write_json_tofile(casedict,metadataoutfile)
            except Exception as err:
                casedict = {"cid":cid,"date":"missing","member":"missing","memberloc":"missing","filenos":"missing","loc":"missing", "sections":"missing","text":"missing","otext":rmnl(s.text),"error":str(err)}
                write_json_tofile(casedict,metadataoutfile)
                continue
        else:
            casedict = {"cid":cid,"date":"missing","member":"missing","memberloc":"missing","filenos":"missing","loc":"missing", "sections":"missing","text":"missing","otext":rmnl(s.text),"error":rmnl(s.text)}
            write_json_tofile(casedict,metadataoutfile)
            continue

def read_htmlfiles(htmlfiles,metadataoutfile):
    """reads the html files as beautifulsoup objects (also identifies cases not collected, or not collected properly by adding an 'ERROR_' prefix to the text"""
    gsoups = []
    for htmlfile in htmlfiles:
        with open(htmlfile, 'r') as f:
            soup = bs(f.read(), 'lxml')
        if len(soup.findAll('div',{'id':'documentMeta'})) >0:
            gsoup= '|'.join([htmlfile,str(soup)])
        else:
            gsoup = '|'.join([htmlfile,"ERROR_"+rmnl(soup.text)])
        gsoups.append(gsoup)
    parse_soups(gsoups,metadataoutfile)
    
"""text preprocessing functions"""
            
def get_filenos(text):
    """get the file numbers from the text"""
    t = text.split('<https:',1)
    fileno = t[0].split('File number',1)
    if len(fileno) > 1:
        fno = t[0].split('File number',1)[1].split('Citation:',1)[0].strip()
        fno = re.sub(':','',fno).strip()
    else:
        fno = 'missing'
    return fno

def get_date(text):
    """get the date of the decision from the text"""
    t = text.split('<https:',1)
    dfl = t[0].split('Date:',1)[1].strip()
    date = dfl.split('File number',1)[0].strip().split('Citation:',1)[0].strip()
    return date

def get_citation(text):
    """get the case citation from the text"""
    t = text.split('<https:',1)
    citation = t[0].split('Citation:',1)[1].strip()
    return citation

def create_splitterdate(row):
    """creates a date object to split the text"""
    d = row['date']
    month = datetime.date(1900, d.month, 1).strftime('%B')
    day = str(d.day)
    year = str(d.year)
    newdate = month + ' ' + day +', '+year
    return newdate

def correct_singlecase(caseid,tshort):
    if caseid =='2015canlii99149':
        t = re.sub('Dated this 9th day of November , 2015','November 9, 2015', tshort)
    else:
        t = tshort
    return t

def get_memberlocdiv(text):
    """use the reference to the date at the end of the text to get metadata on office locations and adjudicators"""
    datesplitter = r"Date Issued|Date Issue|Date Amended|Date Order Issued|Date Reasons were Issued|Date issued|_________________|Original Order|Date Ordered Issued"
    if 'Date Issued' not in text and 'Date Issue' not in text and 'Date Amended' not in text and "Date Order Issued" not in text and "Date Reasons were Issued" not in text and "Date issued" not in text and "_________________" not in text and "Original Order" not in text and "Date Ordered Issued" not in text:
        memberloc = ' '.join([x for x in sent_tokenize(text)[-6:] if "Hearing Officer" in x or "Regional Office" in x or "Member," in x or "Member:" in x])
    else:
        memberloc = re.split(datesplitter,text)[-1].strip()
    return memberloc

def get_member(memberloc):
    """get the adjudicators from the split text (from 'get_memberlocdiv()')"""
    if memberloc !='':
        memberloc = re.sub(',',' ',memberloc).strip()
        member = rmnl(memberloc.split('Landlord and Tenant Board',1)[0].strip())
    else:
        member = 'missing'
    return member

def get_loc(memberloc):
    """get the office locations from the split text (from 'get_memberlocdiv()')"""
    if memberloc !='':
        memberloc = re.sub(',',' ',memberloc).strip()
        loc = memberloc.split('Landlord and Tenant Board',1)#[1].strip()
        if len(loc)==2:
            nloc = rmnl(loc[1].strip())
        else:
            nloc = 'missing'
    else:
        nloc = 'missing'
    return nloc

def preprocess(x,stopwordslist,pattern):
    """preprocess the text, output is used to create the applicant/outcome models"""
    t = x.lower()
    td = t.translate(remove_digits) # removes digits
    ts = pattern.sub('',td) # removes stopwords
    ts = re.sub('\s+',' ',ts).strip() # strips newlines
    ta = re.sub(r'\W+', ' ', ts).strip() # keep only alpha characters
    twords = ta.split(' ')
    twords = [porter.stem(x) for x in twords] # stem words
    twords = [x for x in twords if len(x) >1] # remove words with < 1 character in length
    twords = [x for x in twords if x not in stopwordslist] 
    return ' '.join(twords)

def preprocess_keepnums(x,stopwordslist,pattern):
    """preprocess the text but keep numbers, output is used to create the applicant/outcome models"""
    t = x.lower()
    ts = pattern.sub('',t)
    ts = re.sub('\s+',' ',ts).strip()
    ta = re.sub(r'\W+', ' ', ts).strip()
    twords = ta.split(' ')
    twords = [porter.stem(x) for x in twords]
    twords = [x for x in twords if len(x) >1]
    twords = [x for x in twords if x not in stopwordslist]
    return ' '.join(twords)


def map_cleanjl(x,jldict):
    cleanjl = []
    for k,v in jldict.items():
        x2 = rmnl(re.sub(r'\W+', ' ', x)).lower()
        k2 = rmnl(re.sub(r'\W+', ' ', k)).lower()
        v2 = rmnl(re.sub(r'\W+', ' ', v)).lower()
        if k2 in x2 or v2 in x2:
            cleanjl.append(v)
        else:
            continue
    return '|'.join(list(set(cleanjl)))

def create_stopwordslist():
    monthnames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August','September','October', 'November','December','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct','Nov','Dec', 'Sept']
    stopwordslist = list(get_stop_words('en'))         #About 900 stopwords
    nltk_words = list(stopwords.words('english'))#About 150 stopwords
    monthnames = [x.lower() for x in monthnames]
    stopwordslist.extend(nltk_words)
    stopwordslist.extend(monthnames)
    stopwordslist = list(set([porter.stem(x) for x in stopwordslist]))
    return stopwordslist

def clean_sections(x):
    s = re.sub(r', 2006', ' 2006', x).lower()
    s = rmnl(re.sub('under', '', s).strip().split(' in the matter of',1)[0].strip().split(' review order ',1)[0].strip().split(" m.",1)[0].strip())
    # s = 
    if "residential tenancies act" in s and "powers procedure" in s and s.endswith("and the residential tenancies act 2006"):
        s = rmnl(re.sub("of the statutory powers procedure act and the residential tenancies act 2006"," sppa rta ",s))
    elif "residential tenancies act" in s and "powers procedure" in s and not s.endswith("and the residential tenancies act 2006"):
        s = rmnl(re.sub("of the statutory powers procedure act","sppa",s))
        s = rmnl(re.sub("statutory powers procedure act","sppa",s))
        s = rmnl(re.sub("residential tenancies act 2006","rta",s))
        s = rmnl(re.sub(r"residential tenancies act \(2006\)","rta",s))
        s = rmnl(re.sub("&",',',s))
        s = rmnl(re.sub("of the","",s))
        s = rmnl(re.sub("and section","| section",s))
        s = rmnl(re.sub("and the","|",s))
        s = rmnl(re.sub(" and ",", ",s))
        s = rmnl(re.sub("statutory powers procedures act, 1990","sppa",s))
        s = rmnl(re.sub("statutory powers procedures act","sppa",s))
        s = rmnl(re.sub(r"\| rta , order section","| section",s))
        s = rmnl(re.sub(", order","|",s))
        s = rmnl(re.sub("andorder","|",s))
        s = rmnl(re.sub("ofthe","|",s))
        s = rmnl(re.sub("rta, ","rta|",s))
        s = rmnl(s.split("this amended order",1)[0])
        s = re.sub("amended","",s)
    elif len(x) <100:
        s = rmnl(re.sub("of the statutory powers procedure act","sppa",s))
        s = rmnl(re.sub("statutory powers procedure act","sppa",s))
        s = rmnl(re.sub("residential tenancies act 2006","rta",s))
        s = rmnl(re.sub(r"residential tenancies act \(2006\)","rta",s))
        s = rmnl(re.sub("&",',',s))
        s = rmnl(re.sub("of the","",s))
        s = rmnl(re.sub("and section","| section",s))
        s = rmnl(re.sub("and the","|",s))
        s = rmnl(re.sub(" and ",", ",s))
        s = rmnl(re.sub("statutory powers procedures act, 1990","sppa",s))
        s = rmnl(re.sub("statutory powers procedures act","sppa",s))
        s = rmnl(re.sub(r"\| rta , order section","| section",s))
        s = rmnl(re.sub(", order","|",s))
        s = rmnl(re.sub("andorder","|",s))
        s = rmnl(re.sub("ofthe","|",s))
        s = rmnl(re.sub("rta, ","rta|",s))
        s = rmnl(s.split("this amended order",1)[0])
        s = re.sub("amended","",s)
    else:
        s = ''
    s = rmnl(re.sub(r'[^0-9 ]', '', s))
    return s

""" functions for dealing with cases already collected"""

def get_oldcaseids(folder):
    oldcasefiles = glob.glob(folder)
    oldcaseids = [x.split('/')[-1].split('.html')[0].strip() for x in oldcasefiles]
    return dict(zip(oldcaseids,oldcasefiles))

def get_collected(oldcasefolders):
    oldcasefiles = [get_oldcaseids(x) for x in oldcasefolders]
    oldcasedict = {}
    for i, fileset in enumerate(oldcasefiles):
        if i+1 < len(oldcasefiles):
            for k,v in fileset.items():
                if k in list(oldcasefiles[i+1].keys()):
                    pass
                else:
                    oldcasedict[k] = v
                    continue
        else:
            for k,v in fileset.items():
                oldcasedict[k] = v
            continue
    odf = pd.DataFrame(list(oldcasedict.values()),columns=['oldcasefiles'])
    odf['cid'] = odf['oldcasefiles'].apply(lambda x: x.split('/')[-1].split('.html')[0].strip())
    odf['year'] = odf['cid'].apply(lambda x: int(x.split('canlii')[0]))
    return odf