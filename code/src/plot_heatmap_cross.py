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

def interval_x_delay(df1,df2, target, X_list, remove_list=None, intervals=30, delays=10,p=0,depth=6):
    precisions1418 = {}
    precisions1814 = {}
    for interval in np.arange(10,intervals+1):
        precisions1418[interval] = {}
        precisions1814[interval] = {}

        df_sum1 = load_data_2.sum_df(df1, interval)
        df_sum2 = load_data_2.sum_df(df2, interval)
        for delay in np.arange(1,delays+1)[::-1]:
            W1 = calc_W(df_sum1, target=target, remove_list=None, delay=delay,depth=depth)
            W2 = calc_W(df_sum2, target=target, remove_list=None, delay=delay,depth=depth)
            te_pre1418=predict_by_W(df_sum2, W1, target, delay,depth,plot=False)
            te_pre1814=predict_by_W(df_sum1, W2, target, delay,depth,plot=False)

            precisions1418[interval][delay]= te_pre1418
            precisions1814[interval][delay]= te_pre1814
    return precisions1418,precisions1814

def plot_heatmap(precisions):
    fig, ax = plt.subplots(figsize=(20,8))
    df_prec = pd.DataFrame(precisions)
    df_tmp = df_prec.min()
    cm = generate_cmap(['red','honeydew','blue'])
    sns.heatmap(df_prec,vmin=0.6, vmax=1.2,xticklabels=5,cbar=True,cmap=cm,linewidths=.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(r'$\tau$',fontsize=20)
    plt.xlabel('interval',fontsize=20)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.title('target = '+str(target)+' (CH)',fontsize=30)
    ax.invert_yaxis()
    plt.show()
 



############
target = '05e'
intervals =30
delays = 10
X_list = None
p=0.
depth=2
############

## load data
# df_sum1 : data on day 1
# df_sum2 : data on day 2
df1, df2 = load_data_2.load_data(interval=None, p=p)

precisions1418,precisions1814 = interval_x_delay(df1,df2, target, X_list, remove_list=None, intervals=intervals, delays=delays,p=p,depth=depth)
plot_heatmap(precisions1418)
# plot_heatmap(precisions1814)


