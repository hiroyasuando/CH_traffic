import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# colormapをカスタマイズする
from matplotlib.colors import LinearSegmentedColormap
import load_data0908
from utils0902 import * 


def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def interval_x_delay(df, target, X_list, remove_list=None, intervals=30, delays=10,p=0,param=0.5,th=0.5,depth=6,depth_ar=6):
    precisions = {}
    precisions_ar = {}
    precisions_diff = {}
    for interval in np.arange(10,intervals+1):
        precisions[interval] = {}
        precisions_ar[interval] = {}
        precisions_diff[interval] = {}
        df_sum = load_data0908.sum_df(df, interval)
        for delay in np.arange(1,delays+1)[::-1]:
            _,_, tr_pre, te_pre,_,_,ar_te_pre,_,r2_te_pre,r2_ar_te_pre = predict(df_sum, target=target, remove_list=remove_list, delay=delay,param=param,depth=depth,depth_ar=depth_ar,plot=False)
            # precisions[interval][delay]= te_pre
            # precisions_ar[interval][delay]= ar_te_pre
            precisions[interval][delay]= r2_te_pre
            precisions_ar[interval][delay]= r2_ar_te_pre
            precisions_diff[interval][delay]= te_pre-ar_te_pre
    return precisions,precisions_ar,precisions_diff

def plot_heatmap(precisions):
    fig, ax = plt.subplots(figsize=(20,8))
    #plt.figure(figsize=(20,8))
    df_prec = pd.DataFrame(precisions)
    df_tmp = df_prec.min()
    print(df_prec.min())
    # print(df_tmp.median())
    print(df_prec.idxmin())    
    cm = generate_cmap(['blue','honeydew','red'])
    # cm = generate_cmap(['red','honeydew','blue'])
    sns.heatmap(df_prec,vmin=0., vmax=.7,xticklabels=5,cbar=True,cmap=cm,linewidths=.5,annot=False)
    # sns.heatmap(df_prec,vmin=0.6, vmax=1.2,xticklabels=5,cbar=True,cmap=cm,linewidths=.5,annot=False)
    # sns.heatmap(df_prec,vmin=0, vmax=20,cbar=True,cmap=cm,linewidths=.5)
    # ax.invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(r'$\tau$',fontsize=20)
    plt.xlabel('interval',fontsize=20)
    plt.title('target = '+str(target),fontsize=30)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    # plt.title('target = '+str(target)+' (Reservoir)',fontsize=30)
    #plt.show()
    filename = "hm05e_fl"+ str(depth) +".eps"
    fig.savefig(filename)
 
def plot_heatmap_ar(precisions):
    fig, ax = plt.subplots(figsize=(20,8))
    #plt.figure(figsize=(20,8))
    df_prec = pd.DataFrame(precisions)
    cm = generate_cmap(['blue','honeydew','red'])
    # cm = generate_cmap(['red','honeydew','blue'])
    sns.heatmap(df_prec,vmin=0., vmax=.7,cbar=True,cmap=cm,linewidths=.5,annot=True)
    # sns.heatmap(df_prec,vmin=0.6, vmax=1.2,cbar=True,cmap=cm,linewidths=.5,annot=True)
    # sns.heatmap(df_prec,vmin=0, vmax=20,cbar=True,cmap=cm,linewidths=.5)
    plt.xticks(fontsize=15)
    plt.ylabel('tau',fontsize=20)
    plt.xlabel('interval',fontsize=20)
    # plt.title('target = '+str(target)+' (Reservoir)',fontsize=30)
    # ax.invert_yaxis()
    #plt.show()
    filename = "hmar05e_fl"+ str(depth_ar)  + ".eps"
    fig.savefig(filename)

def plot_heatmap_diff(precisions):
    fig, ax = plt.subplots(figsize=(20,8))
    #plt.figure(figsize=(20,8))
    df_prec = pd.DataFrame(precisions)
    cm = generate_cmap(['red','honeydew','blue'])
    sns.heatmap(df_prec,cbar=True,cmap=cm,linewidths=.5,annot=True)
    # sns.heatmap(df_prec,vmin=0, vmax=20,cbar=True,cmap=cm,linewidths=.5)
    plt.xticks(fontsize=15)
    plt.ylabel('tau',fontsize=20)
    plt.xlabel('interval',fontsize=20)
    # plt.title('target = '+str(target)+' (Reservoir)',fontsize=30)
    ax.invert_yaxis()
    #plt.show()
    filename = "hmdiff05e_fl"+ str(depth_ar)  + ".eps"
    fig.savefig(filename)



############
target = '05e'
intervals =30
delays = 10
X_list = None
p=0.
param=0.5
th=0.4
depth=2
depth_ar=6
############

## データ読み込み
# df_sum1 : 14日のデータ
# df_sum2 : 18日のデータ
df1, df2 = load_data0908.load_data(interval=None, p=p)

precisions,precisions_ar,precisions_diff = interval_x_delay(df1, target, X_list, remove_list=None, intervals=intervals, delays=delays,p=p,param=param,th=th,depth=depth,depth_ar=depth_ar)
plot_heatmap(precisions)
plot_heatmap_ar(precisions_ar)
plot_heatmap_diff(precisions_diff)

