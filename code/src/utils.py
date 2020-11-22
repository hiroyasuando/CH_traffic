import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def NRMSE(y, y_pred):
    return np.sqrt(np.mean((y-y_pred)**2))/np.std(y)


def HamD(y,y_pred,threshold):
    y_bit = np.where(y< threshold, 0, 1)
    y_pred_bit = np.where(y_pred< threshold, 0, 1)
    return np.count_nonzero(y_bit ^ y_pred_bit)
    

def nonlinear(y):
    x = y/np.max(y)
    return x

    

def predict_by_W_rob(df_sum, W, target, delay,depth,plot=False,remove_list=None):
    
    if remove_list != None:
        df_X = df_sum.drop(remove_list, axis=1)
    else:
        df_X = df_sum.drop(target,axis=1)

    for col in df_X.columns:
        df_X[col] = df_X[col] / df_X[col].max()
        for i in range(1,depth):
            df_X[f'{col}_lag_{i}'] = df_X[col].shift(i)
    df_X = df_X.dropna()

    df_y = df_sum[target][depth-1:]

    X = np.array(df_X.iloc[:-delay,:])
    y = np.array(df_y[delay:])
    y = nonlinear(y)

    
    pred = X@W
    pred[pred<0]=0
    tr_len = int(X.shape[0]*0.8)

    if plot:
        print('NRMSE : ',NRMSE(y[-int(X.shape[0]*0.2):],pred[-int(X.shape[0]*0.2):]))
        fig=plt.figure(figsize=(12,3))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(range(tr_len,X.shape[0]),y[-int(X.shape[0]*0.2)-1:], label='target')
        plt.plot(range(tr_len,X.shape[0]),pred[-int(X.shape[0]*0.2)-1:], label='predict')
        plt.legend(bbox_to_anchor=(1.12, 1),fontsize=10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Time',fontsize=20)
        plt.ylabel(r'Normalized $y$',fontsize=20)
        ax.set_ylim(-0.1,1.1)
        plt.show()


    return NRMSE(y[-int(X.shape[0]*0.2):],pred[-int(X.shape[0]*0.2):])

def predict_by_W(df_sum, W, target, delay,depth,plot=False):

    df_X = df_sum.drop(target, axis=1)
    
    for col in df_X.columns:
        df_X[col] = df_X[col] / df_X[col].max()
        for i in range(1,depth):
            df_X[f'{col}_lag_{i}'] = df_X[col].shift(i)
    df_X = df_X.dropna()


    df_y = df_sum[target][depth-1:]

    X = np.array(df_X.iloc[:-delay,:])
    y = np.array(df_y[delay:])
    y = nonlinear(y)

    pred = X@W
    pred[pred<0]=0    

    if plot:
        print('NRMSE : ',NRMSE(y, pred))
        fig=plt.figure(figsize=(12,3))
        plt.title('Half of 5e on Feb. 18',size=20)
        plt.plot(y[-int(X.shape[0]*0.5):], label='target')
        plt.plot(pred[-int(X.shape[0]*0.5):], label='predict')
        plt.legend(bbox_to_anchor=(1.12, 1),fontsize=10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Time',fontsize=20)
        plt.ylabel(r'Normalized $y$',fontsize=20)
        plt.show()


    return NRMSE(y,pred)



def calc_W_rob(df_sum, target, remove_list=None, delay=1,lam=0.05,depth=1,):

    df_X = df_sum.drop(target,axis=1)

    
    for col in df_X.columns:
        df_X[col] = df_X[col] / df_X[col].max()
        for i in range(1,depth):
            df_X[f'{col}lag_{i}'] = df_X[col].shift(i)
    df_X = df_X.dropna()


    df_y = df_sum[target][depth-1:]


    X = np.array(df_X.iloc[:-delay,:])
    y = np.array(df_y[delay:])
    y = nonlinear(y)

    tr_len = int(X.shape[0]*0.8)


    X_tr = X[:tr_len]
    y_tr = y[:tr_len]
    W = (np.linalg.inv(X_tr.T@X_tr + lam*np.identity(X.shape[1])))@X_tr.T@y_tr

    

    l=list(df_X.columns)

    rl=remove_list.copy()
    rl.remove(target)
    ll=[]
    for i in rl:
        l_in = [s for s in l if i in s]
        for j in l_in:
            ll.append(l.index(j))            

    W_del=np.delete(W,ll,0)
    return W_del

def calc_W(df_sum, target, remove_list=None, delay=1,lam=0.05,depth=1,):

    if remove_list != None:
        df_X = df_sum.drop(remove_list, axis=1)
    else:
        df_X = df_sum.drop(target,axis=1)

    

    for col in df_X.columns:
        df_X[col] = df_X[col] / df_X[col].max()
        for i in range(1,depth):
            df_X[f'{col}lag_{i}'] = df_X[col].shift(i)
    df_X = df_X.dropna()


    df_y = df_sum[target][depth-1:]


    X = np.array(df_X.iloc[:-delay,:])
    y = np.array(df_y[delay:])
    y = nonlinear(y)


    W = (np.linalg.inv(X.T@X + lam*np.identity(X.shape[1])))@X.T@y
    
    return W

def predict(df_sum, target, remove_list=None, delay=1,lam=0.05,depth=6,depth_ar=6,plot=False):

    if remove_list != None:
        df_X = df_sum.drop(remove_list, axis=1)
    else:
        df_X = df_sum.drop(target,axis=1)
        

    for col in df_X.columns:
        df_X[col] = df_X[col] / df_X[col].max()
        
        for i in range(1,depth):
            df_X[f'{col}_lag_{i}'] = df_X[col].shift(i)
    df_X = df_X.dropna()

    df_y = df_sum[target][depth-1:]

    X = np.array(df_X.iloc[:-delay,:])
    y = np.array(df_y[delay:])
    y = nonlinear(y)
    
    tr_len = int(X.shape[0]*0.8)


    X_tr = X[:tr_len]
    y_tr = y[:tr_len]
    W = (np.linalg.inv(X_tr.T@X_tr + lam*np.identity(X.shape[1])))@X_tr.T@y_tr


    X_te = X[tr_len:]
    y_te = y[tr_len:]
    y_preds_tr = X_tr@W
    y_preds_tr[y_preds_tr<0]=0    
    y_preds_te = X_te@W
    y_preds_te[y_preds_te<0]=0


    ##########AR MODLE##########
    ddd = df_sum[[target]]
    ddd=ddd.rename(columns={target:'lag_0'})
    for i in range(1,depth_ar):
        ddd[f'lag_{i}'] = ddd['lag_0'].shift(i)
    ddd = ddd.dropna()
    ddd['bias']=1
    ardf_X = ddd
    ardf_y = ddd['lag_0']
    
    arX = np.array(ardf_X.iloc[:-delay,:])
    ary = np.array(ardf_y[delay:])
    ary = nonlinear(ary)

    artr_len = int(arX.shape[0]*0.8)

    arX_tr = arX[:artr_len]
    ary_tr = ary[:artr_len]

    arW = (np.linalg.inv(arX_tr.T@arX_tr + lam*np.identity(arX.shape[1])))@arX_tr.T@ary_tr
    

    arX_te = arX[artr_len:]
    ary_te = ary[artr_len:]
    ary_preds_tr = arX_tr@arW
    ary_preds_te = arX_te@arW
    ary_preds_tr[ary_preds_tr<0]=0
    ary_preds_te[ary_preds_te<0]=0
   ###########

    
    if plot:
        fig=plt.figure(figsize=(12,6))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.subplot(2,1,1)

        ###plot test AR###
        plt.plot(range(artr_len,arX.shape[0]),ary_te, label='target')
        plt.plot(range(artr_len,arX.shape[0]),ary_preds_te, label='predict')
        plt.title('AR NRMSE='+str(np.round(float(NRMSE(ary_te, ary_preds_te)),4)),size=20)
        plt.legend(bbox_to_anchor=(1.0, 1),fontsize=10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # plt.xlabel('Time',fontsize=20)
        plt.ylabel(r'Normalized $y$',fontsize=20)

        

        plt.subplot(2,1,2)
        plt.plot(range(tr_len,X.shape[0]),y_te, label='target')
        plt.plot(range(tr_len,X.shape[0]),y_preds_te, label='predict')
        plt.title(f'CH(remove {remove_list}) NRMSE='+str(np.round(float(NRMSE(y_te, y_preds_te)),4)),size=20)
        # plt.title('CH',size=20)
        plt.legend(bbox_to_anchor=(1., 1),fontsize=10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Time',fontsize=20)
        plt.ylabel(r'Normalized $y$',fontsize=20)


        plt.show()



    return list(df_X.columns), W, NRMSE(y_tr, y_preds_tr), NRMSE(y_te, y_preds_te), list(y_preds_tr)+ list(y_preds_te), list(ary_preds_tr)+list(ary_preds_te),NRMSE(ary_te, ary_preds_te),list(y)


def real_predict(df_sum, target, delay=1, tr_rate=0.8, te_rate=0.2, lam=0.05,depth=1,plot=False):
    df_X = df_sum.drop(target,axis=1)

    for col in df_X.columns:
        df_X[col] = df_X[col] / df_X[col].max()
        for i in range(1,depth):
            df_X[f'{col}_lag_{i}'] = df_X[col].shift(i)
    df_X = df_X.dropna()


    df_y = df_sum[target][depth-1:]


    X = np.array(df_X.iloc[:-delay,:])
    y = np.array(df_y[delay:])
    y = nonlinear(y)

    tr_len_s = math.ceil(X.shape[0]*tr_rate)
    te_len = int(round(tr_len_s/te_rate,0)-tr_len_s)
    
    y_te = y[tr_len_s:]
    y_preds_te = np.zeros(y_te.shape)
    
    for i in range(int((X.shape[0] - tr_len_s)/te_len)+1):
        tr_len = tr_len_s + te_len*i
        X_te = X[tr_len:tr_len+te_len]
        
        if tr_len > X.shape[0]:
            tr_len = X.shape[0]
            X_te = X[tr_len:]
        

        X_tr = X[:tr_len]
        y_tr = y[:tr_len]
        W = (np.linalg.inv(X_tr.T@X_tr + lam*np.identity(X.shape[1])))@X_tr.T@y_tr

        y_preds_tr = X_tr@W
        y_pred = X_te@W
        y_pred[y_pred<0]=0
        try:
            y_preds_te[te_len*i:te_len*(i+1)] = y_pred
        except:
            y_preds_te[te_len*i:] = y_pred

    if plot:
        fig=plt.figure(figsize=(12,3))
        plt.plot(y_te[-int(X.shape[0]):], label='target')
        plt.plot(y_preds_te[-int(X.shape[0]):], label='predict')
        plt.ylim(0,1.5)
        plt.title(r'$r_1=0.5,r_2=0.9$',size=20)
        plt.legend(bbox_to_anchor=(1.13, 1),fontsize=10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Time',fontsize=20)
        plt.ylabel(r'Normalized $y$',fontsize=20)
        plt.show()

        
    return NRMSE(y_tr, y_preds_tr), NRMSE(y_te, y_preds_te),

def real_time_predict(df_sum, target,  delay,depth):
    precisions = {}
    for tr_rate in np.arange(0.1, 1.0, 0.1):
        tr_rate = round(tr_rate, 1)
        precisions[tr_rate] = {}

        for te_rate in (1-np.arange(0.1, 1-tr_rate+1e-10, 0.1)):
            te_rate = round(te_rate, 2)
            if te_rate==0.9 and tr_rate==0.2: 
                tr_prec, te_prec = real_predict(df_sum, target=target, delay=delay, tr_rate=tr_rate, te_rate=te_rate,depth=depth,plot=False)
            elif te_rate==0.2 and tr_rate==0.2:
                tr_prec, te_prec = real_predict(df_sum, target=target, delay=delay, tr_rate=tr_rate, te_rate=te_rate,depth=depth,plot=False)
            else:
                tr_prec, te_prec = real_predict(df_sum, target=target, delay=delay, tr_rate=tr_rate, te_rate=te_rate,depth=depth,plot=False)
            precisions[tr_rate][te_rate] = te_prec
            
    df_prec = pd.DataFrame(precisions)

    cm = generate_cmap(['red','honeydew','blue'])
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df_prec,vmin=0.6, vmax=1.2,cbar=True,cmap=cm,linewidths=.5, annot=True)
    ax.invert_yaxis()

    plt.ylabel(r'Rate of Train Length: $r_2$',fontsize=20)
    plt.xlabel(r'Rate of Train lenght of Total Length: $r_1$',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    plt.show()

def ts_fft(df_sum, n, col, interval):

    for i in reversed(range(10)):
        x = int(len(df_sum[col]) / 2**i)
        if x >= 1:
            n = i
            break
            

    ts = np.array(df_sum[col][:2**n]) - np.mean(df_sum[col][:2**n])
    F = np.fft.fft(ts)
    F_abs = np.abs(F)
    F_abs_amp = F_abs / (2**n)*2
    F_abs_amp[0] = F_abs_amp[0] / 2
    freq = np.fft.fftfreq(2**n,d=interval) 

    
    return F_abs_amp,freq

def get_n_Cols_Fs(df_sum, target, roads, w,depth,interval):


    df_X=df_sum
    for col in df_X.columns:
        df_X[col] = df_X[col] / df_X[col].max()
        
        for i in range(1,depth):
            df_X[f'{col}_lag_{i}'] = df_X[col].shift(i)
    df_X = df_X.dropna()

    for i in reversed(range(10)):
        x = int(len(df_X) / 2**i)
        if x >= 1:
            n = i
            break

    w_abs = np.abs(w)
    cols = [target]
    Ws = ['target']
    for i in range(1,6):
        cols.append(roads[np.where(w_abs==np.sort(w_abs)[-i])[0][0]])
        Ws.append(w[np.where(w_abs==np.sort(w_abs)[-i])[0][0]])


    Fs = []
    freqs = []
    for col in cols:
        F ,freq= ts_fft(df_X, n, col,interval)
        Fs.append(F)
        freqs.append(freq)


    return n, cols, Fs, Ws,freqs

def generate_cmap(colors):

    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)



def plot_period(df, cols, Fs, Ws, freqs,interval, n,depth):

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(wspace=0.1, hspace=1)
    i = 1
    for col, F, w, freq in zip(cols, Fs, Ws,freqs):
        ax = fig.add_subplot(len(cols), 2, i)
        if w == 'target':
            ax.set_title(col+' (w = '+str(w)+')',fontsize=10)
        else:
            ax.set_title(col+' (w = '+str(np.round(float(w),4))+')',fontsize=10)

        ax.plot(df[col][:2**(n-1)+1])
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        if i==2*len(cols)-1:
            ax.set_xlabel('time', fontsize=10)

        ax1 = fig.add_subplot(len(cols), 2, i+1)

        ax1.plot(freq[:2**(n-1)],F[:2**(n-1)])

        ax1.tick_params(axis='x', labelsize=10)
        ax1.tick_params(axis='y', labelsize=10)
        ax1.set_xticks(ax1.get_xticks()[1::2])

        if i==2*len(cols)-1: 
            ax1.set_xlabel('frequency',fontsize=10)
        i += 2
    plt.show()
