import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import load_data_2
from utils import * 


def generate_cmap(colors):
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def interval_x_delay(df, target, X_list, remove_list=None, intervals=30, delays=10,p=0,depth=6,depth_ar=6):
    precisions = {}
    precisions_ar = {}
    precisions_diff = {}
    for interval in np.arange(10,intervals+1):
        precisions[interval] = {}
        precisions_ar[interval] = {}
        precisions_diff[interval] = {}
        df_sum = load_data_2.sum_df(df, interval)
        for delay in np.arange(1,delays+1)[::-1]:
            _,_, tr_pre, te_pre,_,_,ar_te_pre,_ = predict(df_sum, target=target, remove_list=remove_list, delay=delay,depth=depth,depth_ar=depth_ar,plot=False)
            precisions[interval][delay]= te_pre
            precisions_ar[interval][delay]= ar_te_pre
    return precisions,precisions_ar

def plot_heatmap(precisions):
    fig, ax = plt.subplots(figsize=(20,8))
    df_prec = pd.DataFrame(precisions)
    df_tmp = df_prec.min()
    cm = generate_cmap(['red','honeydew','blue'])
    sns.heatmap(df_prec,vmin=0.6, vmax=1.2,xticklabels=5,cbar=True,cmap=cm,linewidths=.5,annot=False)
    ax.invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(r'$\tau$',fontsize=20)
    plt.xlabel('interval',fontsize=20)
    plt.title('target = '+str(target)+'(CH)',fontsize=30)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.show()

 
def plot_heatmap_ar(precisions):
    fig, ax = plt.subplots(figsize=(20,8))
    df_prec = pd.DataFrame(precisions)
    cm = generate_cmap(['red','honeydew','blue'])
    sns.heatmap(df_prec,vmin=0.6, vmax=1.2,cbar=True,cmap=cm,linewidths=.5,annot=False)
    plt.xticks(fontsize=15)
    plt.ylabel('tau',fontsize=20)
    plt.xlabel('interval',fontsize=20)
    plt.title('target = '+str(target)+'(AR)',fontsize=30)
    ax.invert_yaxis()
    plt.show()




############
target = '05e'
intervals =30
delays = 10
X_list = None
p=0.
depth=2
depth_ar=6
############

## loada data
# df_sum1 : data on day1
# df_sum2 : data on day2
df1, df2 = load_data_2.load_data(interval=None, p=p)

precisions,precisions_ar = interval_x_delay(df2, target, X_list, remove_list=None, intervals=intervals, delays=delays,p=p,depth=depth,depth_ar=depth_ar)
plot_heatmap(precisions)
# plot_heatmap_ar(precisions_ar)

