import pandas as pd
import numpy as np
from jqdatasdk import *
auth('13310*****','password')

def fullcode(code):
    xcode=code[-6:]+'.XSHG' if code[-6]=='6' else code[-6:]+'.XSHE'
    return xcode

def get_holdings(code,start='2019-01-01'):
    '''
    股东户数
    '''
    q=query(finance.STK_HOLDER_NUM).filter(finance.STK_HOLDER_NUM.code==fullcode(code),finance.STK_HOLDER_NUM.pub_date>start)
    df=finance.run_query(q) 
    return df


def get_holder_counts(code,start='2019-01-01'):
    '''
    股东户数
    '''
    q=query(finance.STK_HOLDER_NUM).filter(finance.STK_HOLDER_NUM.code==fullcode(code),finance.STK_HOLDER_NUM.pub_date>start)
    df=finance.run_query(q) 
    return df


def get_holder_change(code,start='2019-01-01'):
    '''
    增减持
    '''     
    q=query(finance.STK_SHAREHOLDERS_SHARE_CHANGE).filter(finance.STK_SHAREHOLDERS_SHARE_CHANGE.code==fullcode(code),\
                                                          finance.STK_SHAREHOLDERS_SHARE_CHANGE.pub_date>start)
    df=finance.run_query(q)
    return df

def get_limited_open(code,start='2019-01-01'):
    '''
    限售股解禁
    '''    
    q=query(finance.STK_LIMITED_SHARES_UNLIMIT).filter(finance.STK_LIMITED_SHARES_UNLIMIT.code==fullcode(code),\
                                                              finance.STK_LIMITED_SHARES_UNLIMIT.pub_date>start)
    df=finance.run_query(q)
    return df

