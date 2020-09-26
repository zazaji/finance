import numpy as np
import pandas as pd
import tushare as ts
import baostock as bs
import time

def ltsmmodel(xshape,yshape,model_class='classifier'):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from keras.models import Sequential
    from keras.layers import LSTM,Dense, Dropout, Flatten,Activation
    from keras.layers.normalization import BatchNormalization
    model=Sequential()
    model.add(LSTM(units=400,input_shape=(xshape[1],xshape[2]),return_sequences=True))  ##activation='softmax', 
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(LSTM(units=200,return_sequences=True))  ##activation='softmax', 
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))    
    model.add(LSTM(units=100)), 
    model.add(Dropout(0.5))
    model.add(Dense(units=yshape))
    if model_class=='classifier':
        #分类预测
        model.add(Activation('softmax')) # 输出各类的概率(softmax)
        model.compile(optimizer='adam', loss=mycrossentropy,metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
        # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', mean_pred])  
    else:
        #数值预测
        model.add(Activation('linear'))
        model.compile(optimizer='adam',loss='mean_absolute_error')
    return model



def mycrossentropy(y_true, y_pred,nb_classes=10, e=0.1):
    return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/nb_classes)
    



def LSTMcutdatas(data_x,data_y,sequence_length):
    data_x_new ,data_y_new= [],[]
    for i in range(len(data_x) - sequence_length+1):
        rangedata=data_x[i: i + sequence_length]
        data_x_new.append(rangedata)
    for y in data_y:
        data_y_new.append(np_utils.to_categorical(y[sequence_length-1:]))
    data_x_new = np.array(data_x_new)

    return data_x_new,data_y_new



    def indexpredict(symbol='sh.000001',gate=[2,3,4,5],ktype='d',m='1',trshold=0.3,testlen=100):
    mbins=10
    from sklearn import ensemble
    
    today=time.strftime('%Y-%m-%d',time.localtime(time.time()))
    data=getBaoStocksPrice(symbol,today,start_date='2015-10-01')
    lstm=1 if m=='4' or m==4 else 0
    X_train,y_trains,X_test,y_tests,X_pred,close=preDealIndexData(data,lstm=lstm,testlen=100,mbins=10)

    if m=='0':
        import lightgbm as lgb
        model=lgb.LGBMClassifier()

    elif m=='1':
        from xgboost import XGBClassifier
        model=XGBClassifier(learning_rate=0.005, n_estimators=300,         # 树的个数--500棵树建立xgboost
                              scale_pos_weight=1,        # 惩罚权重，解决样本个数不平衡的问题
                              max_depth=20, 
                              objective='multi:softmax',
                              min_child_weight=1,      # 叶子节点最小权重
                              gamma=0.5,                  # 惩罚项中叶子结点个数前的参数
                              subsample=0.6,             # 随机选择60%样本建立决策树
                              colsample_btree=0.6,       # 随机选择60%特征建立决策树
                              random_state=2,           # 随机数
                              n_jobs=-1)

    elif m=='2':
        model=ensemble.RandomForestClassifier(n_estimators=400,max_depth=30, criterion='gini')

    elif m=='3':
        from sklearn.svm import SVC
        model=SVC(C=1, kernel='rbf', gamma='auto',probability=True)
    elif  m=='4':
        # model=ltsmmodel(X_train.shape,mbins)
        from keras.models import load_model


    X_train[np.isnan(X_train)]=0.5
    all_probas,probas,predictions, proba_all,y_strs=[],[],[],[],[]
    plus=np.array([-4.8,-4,-3,-2,0,0,2,3,4,4.8])
    X_test=np.r_[X_test,X_pred]
    for i in range(2):
        if lstm==1:
            model = load_model('/www/finance/albumy/stock/model.h5',custom_objects={'mycrossentropy': mycrossentropy,})
            model.fit(X_train,y_trains[i],epochs=2,batch_size=64,verbose=1)
            y_pred=model.predict(X_test)
            del model
            predictions.append(np.argmax(y_pred,axis=1))
            all_probas.append((y_pred*plus).sum(axis=1))
            probas.append(y_pred[-1])
        else:
            model.fit(X_train,y_trains[i])
            predictions.append(model.predict(X_test))
            y_pred=model.predict_proba(X_test)
            all_probas.append((y_pred*plus).sum(axis=1))
            probas.append(y_pred[-1])
        y_strs.append('d'+str(i+1))

    probas=pd.DataFrame(np.array(probas).T,columns=y_strs)
    probas['ii']=range(len(probas))    

    for i in range(len(y_strs)):
        proba_all.append({'data': np.array(probas[['ii',y_strs[i]]]).tolist(),'name':y_strs[i]})

    y_r=pd.DataFrame(np.zeros(len(predictions[0])),columns=['bsig'])

    # y_r.loc[(predictions[0]+2*predictions[1]<=gate[0]+2*gate[1])|(all_probas[0]+all_probas[1]<-0.5),'bsig']=-1
    # y_r.loc[(predictions[0]+2*predictions[1]>=gate[2]+2*gate[3])&(all_probas[0]+all_probas[1]>0.4),'bsig']=1
    if lstm==1:
        y_r.loc[(all_probas[0]+all_probas[1]<(gate[0]+gate[1])-9)/3.3),'bsig']=-1
        y_r.loc[(all_probas[0]+all_probas[1]>(gate[2]+gate[3])-9)/3.3),'bsig']=1
    else:
        y_r.loc[(predictions[0]<=gate[0])&(predictions[1]<=gate[1]),'bsig']=-1
        y_r.loc[(predictions[0]>=gate[2])&(predictions[1]>=gate[3]),'bsig']=1
        # y_r.loc[(predictions[0]>=gate[2])&(predictions[1]>=gate[3]),'bsig']=1
        # finas=calcReturns(close.values,[predictions,predictions2],thrd=[1,5])
    p_predict_avg=10*np.ones(len(close.values))
    states_buy,states_sell,total_gains,invest,gains,percent_holding,operations \
            = buy_stock(close, y_r['bsig'].values[-len(close):],100000000,100,100,0.8,0.8,0,p_predict_avg)
    finas=showpicToCode(close.values,states_buy,states_sell,gains,percent_holding,total_gains, invest,np.array(predictions))

    # finas=imgToCode(finas)
    return data,y_tests,predictions,finas,proba_all#,y_r['bsig'].values.tolist()

    
def preDealIndexData(result,lstm=0, testlen=50,mbins=10,sequence_length=5,endlen=3):
    import albumy.stock.newfeatureH as newfeatureH
    X_col='pctChg', 'openjump', 'closedown', 'daydown', 'highup', 'lowdown',\
       'openupup', 'closeupup',  'p_change','ma5angle', 'ma20angle', 'volratio5', 'volratio5_40'
#           'todayup', 'todayopenup',\

    y_col=['cpct_1_c','cpct_2_c','cpct_3_c']
    result=newfeatureH.get_result(result)
    resultclose=result['close'].copy()
    result=result[50:]
    result=newfeatureH.norm_data(result,stds=2)
    result['close']=resultclose
    # X=pd.merge(result[[col for col in X_col]],X,left_index=True,right_index=True,how='left')
    X=result[[col for col in X_col]].values
    y=result[[col for col in y_col]].fillna(5)[:-endlen]
    y_values=[]
    for col in y_col:
        ya=y[col]
        bins=np.unique(1*np.r_[-1e20,[np.round(np.quantile(ya, i/mbins),5) for i in range(1,mbins)],1e20])
        ya=pd.cut(ya,bins=bins,labels=False)
        y_values.append(ya)
    if lstm==1:
        X,y_values=LSTMcutdatas(X,y_values,sequence_length)


    X_train=X[:-testlen-endlen]
    X_test=X[-testlen-endlen:-endlen]
    X_pred=X[-endlen:]
    y_trains,y_tests=[],[]
    for i in range(len(y_col)):
        y_trains.append(y_values[i][:-testlen])
        y_tests.append(y_values[i][-testlen:])
    return X_train,y_trains,X_test,y_tests,X_pred,result['close'][-testlen:]
    

if __name__ == '__main__':
    '''
    回测
    '''
    pass
