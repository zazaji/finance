import numpy as np
import pandas as pd
import tushare as ts
import baostock as bs
import time


#######获取指数数据###########
def getIndexD(symbol, start='2005-01-01', end='2030-10-10', ktype='d'):
    infields="date,open,high,low,close,volume,amount,turn"
    rs = bs.query_history_k_data_plus(symbol, infields,start_date=start, end_date=end,
                                    frequency=ktype, adjustflag="3")
       
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    result=result[result['date']>start]
    if len(result)>0:
        result=result[result['amount']!='0']
        result=result[result['amount']!='']
        result=result.set_index(['date']).astype('float32').sort_index()
        result['volume']=result['volume'].astype('float32')/1e8
        result['amount']=result['amount'].astype('float32')/1e8
        result['turn']=result['turn']*100
        result=result
        return result.reset_index()
    else:
        return []
    
def getStockD(symbol, start='2005-01-01', end='2030-10-10', ktype='d',index=0):
    infields="date,open,high,low,close,preclose,volume,amount,turn,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"
    rs = bs.query_history_k_data_plus(symbol, infields,start_date=start, end_date=end,
                                    frequency=ktype, adjustflag="2")

    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result=result[result['date']>start]
    if len(result)>0:
        result=result[result['amount']!='0']
        result=result[result['amount']!='']
        result['volume']=result['volume'].astype('float32')/1e7
        result['amount']=result['amount'].astype('float32')/1e8
        result=result[result['amount']>0]
        result.loc[result['isST']=='','isST']=0
        result.loc[result['peTTM']=='','peTTM']=-1
        result.loc[result['psTTM']=='','psTTM']=-1
        result.loc[result['pbMRQ']=='','pbMRQ']=-1
        result.loc[result['pcfNcfTTM']=='','pcfNcfTTM']=-1
        result=result.set_index(['date']).astype('float32').sort_index()
        result['isST']=result['isST'].astype('int8')

        return result.reset_index()
    else:
        return []

def getStockH(symbol, start='2015-01-01', end='2020-10-25', ktype='60',index=0):
    infields="time,open,high,low,close,volume,amount"
    rs = bs.query_history_k_data_plus(symbol, infields,start_date=start, end_date=end,
                                    frequency=ktype, adjustflag="2")
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    if len(result)>50:
        result=result.sort_values('time')
        result['date']=result['time'].str[:8]
        result['hour']=result['time'].str[8:12].astype('int16')
        result=result[result['volume']!='0']
        result=result[result['volume']!='']
        result=result[result['amount']!='0']
        result=result[result['amount']!='']
        result['volume']=result['volume'].astype('float32')/1e6
        result['amount']=result['amount'].astype('float32')/1e7
        result=result.set_index(['date','hour']).drop('time',axis=1).astype('float32').sort_index()
        return result 
    else:
        return []   
    
def get_quarter_data(code,start='2019-01-01',end='2020-02-02'):
    start_year=start[:4]
    end_year=end[:4]
    cash_flow_list=[]
    profit_list = []
    balance_list =[]
    growth_list = []

    for year in range(int(start_year),int(end_year)):
        for quarter in range(1,5):
            if today[:4]>=str(year):
#                 print(code,start, end)
                rs_cash_flow = bs.query_cash_flow_data(code=code, year=year, quarter=quarter)##现金流
                while (rs_cash_flow.error_code == '0') & rs_cash_flow.next():
                    cash_flow_list.append(rs_cash_flow.get_row_data())
                rs_profit = bs.query_profit_data(code=code, year=year, quarter=quarter)##盈利
                while (rs_profit.error_code == '0') & rs_profit.next():
                    profit_list.append(rs_profit.get_row_data())
                rs_balance = bs.query_balance_data(code=code, year=year, quarter=quarter) ##资产负债
                while (rs_balance.error_code == '0') & rs_balance.next():
                    balance_list.append(rs_balance.get_row_data())
                rs_growth = bs.query_growth_data(code=code, year=year, quarter=quarter) ##成长
                while (rs_growth.error_code == '0') & rs_growth.next():
                    growth_list.append(rs_growth.get_row_data())
                    
    result_cash_flow = pd.DataFrame(cash_flow_list, columns=rs_cash_flow.fields)
    result_profit = pd.DataFrame(profit_list, columns=rs_profit.fields)
    result_balance = pd.DataFrame(balance_list, columns=rs_balance.fields)
    result_growth = pd.DataFrame(growth_list, columns=rs_growth.fields)

    return result_cash_flow,result_profit,result_balance,result_growth

def get_forcast(code,start='2019-01-01',end='2020-02-02'):
    result_list = []
    rs_forecast_list = []
#     print(code,start, end)
    rs = bs.query_performance_express_report(code, start_date=start, end_date=end)
    while (rs.error_code == '0') & rs.next():
        result_list.append(rs.get_row_data())
    result = pd.DataFrame(result_list, columns=rs.fields)
    
    rs_forecast = bs.query_forecast_report(code, start_date=start, end_date=end)
    while (rs_forecast.error_code == '0') & rs_forecast.next():
        rs_forecast_list.append(rs_forecast.get_row_data())
    result_forecast = pd.DataFrame(rs_forecast_list, columns=rs_forecast.fields)
    
    return result,result_forecast


    
def getStockList(tradedate):
    '''
    获取股票列表
    '''
    stock_rs = bs.query_all_stock(tradedate)
    stock_df = stock_rs.get_data()
    stock_df=stock_df[(stock_df.code_name.str[:3]!='688')]
    stocks=stock_df[
        (stock_df.code.str[:5]=='sh.60')|
        (stock_df.code.str[:5]=='sh.68')|
        (stock_df.code.str[:5]=='sz.30')|
        (stock_df.code.str[:5]=='sz.00')
               ].code.values
    indexs=stock_df[
        (stock_df.code.str[:5]!='sh.60')&
        (stock_df.code.str[:5]!='sh.68')&
        (stock_df.code.str[:5]!='sz.30')&
        (stock_df.code.str[:5]!='sz.00')
               ].code.values
    return stocks,indexs

def getBaoStocksPrice(symbol,end_date,start_date='2015-10-01'):
    '''
    获取单支股票日K线数据
    '''
    lg = bs.login()
    rs = bs.query_history_k_data_plus(symbol,
        "date,open,high,low,close,preclose,volume,amount,pctChg",
        start_date=start_date, end_date=end_date, frequency="d",adjustflag="2")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    bs.logout()
    return result.set_index('date').astype('float32')


def tradedate():
    '''
    获取最近交易日
    '''
    import datetime
    import time
    yestoday30=(datetime.date.today() + datetime.timedelta(-10)).strftime("%Y-%m-%d") 
    today=time.strftime('%Y-%m-%d',time.localtime(time.time()))

    rs = bs.query_trade_dates(start_date=yestoday30, end_date=today)
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result=result.loc[result.is_trading_day=='1','calendar_date'].values[-2]
    return result


if __name__ == '__main__':
'''
将数据抓取到H5文件
如果全部执行完成需要1天时间，可以设置晚上完成 '''
    today=time.strftime('%Y-%m-%d',time.localtime(time.time()))
    s_date='2009-10-01'
    path='data/'


    bs.login()

    last_traday=tradedate()

    #抓取股票列表，并区分股票和指数
    stocks,indexs=getStockList(last_traday)

    #抓取指数数据
    for i,code in enumerate(indexs[:]):
        result=getIndexD(code,start=str(s_date),end=today)
        if i%100==0:print(i,code,':',len(result))
        if len(result)>0:
            result.to_hdf(path+code+'_d.h5',key='data')

    #抓取股票日线数据
    for i,code in enumerate(stocks[:]):
        result=getStockD(code,start=s_date,end=today)
        if i%100==0:print(i,code,':',len(result))
        if len(result)>0:
            result.to_hdf(path+code+'_d.h5',key='data')

    #抓取分钟线数据
    for i,code in enumerate(stocks[:]):
        for ktype in ['60','15','5']:
            result=getStockH(code,start=s_date,end=today,ktype=ktype)
        #     print(result.head())
            if i%100==0:print(i,code,':',len(result))
            if len(result)>0:
                result.to_hdf(path+code+'_'+ktype+'.h5',key='data')   

    #抓取利润、现金流、等基本面数据
    for i,code in enumerate(stocks[:]):
        try:
            result_cash_flow,result_profit,result_balance,result_growth=get_quarter_data(code,start=s_date,end=today)

            if len(result_cash_flow)>0:result_cash_flow.to_hdf(path+code+'_cash_flow.h5',key='data')
            if len(result_profit)>0:result_profit.to_hdf(path+code+'_'+'result_profit.h5',key='data')
            if len(result_balance)>0:result_balance.to_hdf(path+code+'_'+'result_balance.h5',key='data')
            if len(result_growth)>0:result_growth.to_hdf(path+code+'_'+'result_growth.h5',key='data')

            fast_report,result_forecast=get_forcast(code,start=s_date,end=today)

            if len(fast_report)>0:fast_report.to_hdf(path+code+'_'+'fast_report.h5',key='data')
            if len(result_forecast)>0:result_forecast.to_hdf(path+code+'_'+'result_forecast.h5',key='data')

            if i%10==0:print(i,code,len(result_cash_flow),len(result_profit),len(result_balance),len(result_growth),len(fast_report),len(result_forecast))
        except:
            print('error',code)

    bs.logout()