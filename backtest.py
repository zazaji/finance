import numpy as np
import pandas as pd
import tushare as ts
import baostock as bs
import time

def buy_stock(price,signal,initial_money = 1000000, max_buy = 1000,max_sell =1000,
            buyPct=10, selPct=10, trade_min=100, show=0 ):  # 用预测值来控制仓位
    '''
    回测函数
    price，股价,numpy数组
    signal，加仓1、减仓-1、持有0
    buyPct，selPct一次买入/卖出最小仓位，10%，20%等
    最小买卖股数
    show 是否打印买卖日志
    p_predict_avg 根于预测值控制仓位
    '''
    # price=price.values
    # signal=signal.values
    """
    price = actual movement in the real world
    delay = how much interval you want to delay to change our decision from buy to sell, vice versa
    initial_state = 1 is buy, 0 is sell
    initial_money = 1000, ignore what kind of currency
    max_buy = max quantity for share to buy
    max_sell = max quantity for share to sell
    """
    start_money = initial_money,
    starting_money = initial_money
    states_sell = []
    states_buy = []
    percent_holding=[]
    current_inventory = 0
    gains=[]
    operations=[]
    
    ######买入函数
    def buy(i, initial_money, current_inventory):  #i=时间，initial_money=现金余额，#current_inventory总股票数量
        shares =trade_min*(((initial_money+current_inventory*price[i])*buyPct// (trade_min*price[i])))
        if shares < 1 :
            buy_units=0
            if show==1:
                print(
                    '==========day %d: buy signal, no money , total cash %9.2f,do nothing,now cash %9.1f, holding  %d, mkt %9.1f,============'
                      % (i,initial_money, current_inventory, current_inventory* price[i]+initial_money)

                )
        else:
            max_buy=100*(initial_money //  (trade_min*price[i]))
#             print(shares,max_buy,buyPct,price[i])
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            initial_money -= buy_units * price[i]
            current_inventory += buy_units
            if show==1:
                print(
                    'day %d: buy %d units at money %9.1f, now cash %9.1f, holding  %d, mkt %9.1f, total %9.1f'
                    % (i,buy_units, buy_units * price[i], initial_money,current_inventory, current_inventory* price[i], current_inventory* price[i]+initial_money)
                )
            states_buy.append(0)
        return initial_money,current_inventory,buy_units,buy_units*price[i]
    
    ########卖出函数
    def sell(i, initial_money, current_inventory):        
        if current_inventory == 0 :
            sell_units=0
            if show==1:
                print('==========day %d: sell signal,invest empty,do nothing,now cash %9.1f, holding  %d, mkt %9.1f,============'
                      % (i,initial_money, current_inventory* price[i], current_inventory* price[i]+initial_money)
                     )
        else:
            max_sell=100*((initial_money+current_inventory*price[i])*selPct // (trade_min*price[i]))
            if current_inventory > max_sell:
                sell_units = max_sell
            else:
                sell_units = current_inventory

            current_inventory -= sell_units
            total_sell = sell_units * price[i]
            initial_money += total_sell
            try:
                invest = (
                    (price[i] - price[states_buy[-1]])
                    / price[states_buy[-1]]
                ) * 100
            except:
                invest = 0
            if show==1:
                print('day %d, sell %d units at money %9.1f, now cash %9.1f, holding %d, mkt %d,total  %9.1f, profit %9.1f %% '
                    % (i,sell_units, total_sell, initial_money,current_inventory, current_inventory* price[i], current_inventory* price[i]+initial_money, invest)
                     )
            operations.append([i, price[i], -sell_units, -total_sell, invest, initial_money,current_inventory* price[i]+initial_money])
        states_sell.append(i)
        stockmoney=current_inventory* price[i]
        return initial_money, current_inventory,sell_units,sell_units*price[i]
            
            
    ######
#     for i in range(price.shape[0] - int(0.025 * len(df))):
    for i in range(len(signal)):
#         print(signal)
        state = signal[i]
        if state == 1:###买入条件
            initial_money, current_inventory,buy_units,buy_money = buy(i, initial_money, current_inventory)
            states_buy.append(i)
            operations.append([i,price[i],buy_units,buy_money , 0, initial_money,current_inventory, current_inventory* price[i]+initial_money])
        elif state == -1:###卖出条件
            initial_money, current_inventory,sell_units,buy_money = sell(i, initial_money, current_inventory)
            states_sell.append(i)
            operations.append([i,price[i],sell_units,buy_money , 0, initial_money,current_inventory, current_inventory* price[i]+initial_money])
        else:###不操作
            operations.append([i, price[i],0, 0,0, initial_money,current_inventory, current_inventory* price[i]+initial_money])            
            
            

        gains.append((current_inventory* price[i]+initial_money)/start_money-1)
        pct_holding=(current_inventory* price[i])/(current_inventory* price[i]+initial_money)
#         print(current_inventory,price[i],initial_money)
        percent_holding.append(pct_holding)
        #####计算最终收益
    invest = ((current_inventory* price[i]+initial_money- starting_money) / starting_money) * 100 
    total_gains = initial_money - starting_money+current_inventory* price[i]
    operations=pd.DataFrame(operations,columns=['day','close','buy/sell', 'amount', 'invest', 'money','hoding', 'allmoney'])
    return states_buy, states_sell, total_gains, invest,gains,percent_holding,operations


# 画图
def showpicToCode(close,states_buy,states_sell,gains,percent_holding,total_gains, invest,predictions):
    plt.clf()
    fig = plt.figure(figsize = (12,8))
    
    plt.subplot(411)
    plt.plot(close)
    plt.plot(close, '^', markersize=10, color='g', label = 'buying signal', markevery = states_buy)
    plt.plot(close, 'x', markersize=10, color='r', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %9.2f, total investment %9.2f%%'%(total_gains/1000, invest))
    plt.legend()

    plt.subplot(412)
    plt.plot(gains,c='b',label='cash')
    plt.title('gains VS stock')
    plt.plot(close/close[0]-1,c='y',label='stock')
    plt.legend()

    plt.subplot(413)
    for i,predict in enumerate(predictions):
        plt.plot(predict,label=str(i))
    plt.legend()

    plt.subplot(414)
    plt.title('Positions %9.2f, ret/Positions  %9.2f'%(sum(percent_holding),100*invest/sum(percent_holding)))
    plt.plot(percent_holding)
    buffer = BytesIO()
    plt.savefig(buffer,bbox_inches = 'tight')
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data) 
    ims = imb.decode()
    imd = "data:image/png;base64," + ims  
    return imd 

if __name__ == '__main__':
    '''
    回测
    '''
    pass
