import pandas as pd
import numpy as np
import load_data
from utils import * 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import warnings
warnings.filterwarnings('ignore')


def plot_attractor(df_sum, target, delay,use_wcols=3,tau=1,depth=1,depth_ar=5,method='line',title='11e',plot=False):

    roads, w, _, _,pred, pred_ar,_,ts = predict(df_sum, target=target, remove_list=None, delay=delay,lam=0.05,depth=depth,depth_ar=depth_ar,plot=False)
    df_W = pd.DataFrame({'X':roads, 'W':w, 'W_abs': np.abs(w)})
    

    ddd = pd.DataFrame()
    ddd['lag_0'] = ts
    for i in range(1,5):
        ddd[f'lag_{i}'] = ddd['lag_0'].shift(tau*i)
    ddd = ddd.dropna()


    dd = pd.DataFrame()
    dd['lag_0'] = pred
    for i in range(1,5):
        dd[f'lag_{i}'] = dd['lag_0'].shift(tau*i)
    dd = dd.dropna()


    dd_ar = pd.DataFrame()
    dd_ar['lag_0'] = pred_ar
    for i in range(1,5):
        dd_ar[f'lag_{i}'] = dd_ar['lag_0'].shift(tau*i)
    dd_ar = dd_ar.dropna()
    

    ##Output
    dd.to_csv("./reconstruct.csv",columns=['lag_0','lag_1','lag_2'],header=False,index=False,sep='\t')
    ddd.to_csv("./target.csv",columns=['lag_0','lag_1','lag_2'],header=False,index=False,sep='\t')
    dd_ar.to_csv("./ar.csv",columns=['lag_0','lag_1','lag_2'],header=False,index=False,sep='\t')
   
    ### Plot
    plt.figure(figsize=(24,24))
    fig = make_subplots(rows=1, cols=3,
                    specs=[[{'is_3d': True}, {'is_3d': True},{'is_3d': True}]],
                    print_grid=False,
                   subplot_titles=(target, 'CH','AR'))
    fig.append_trace(
                go.Scatter3d(
                    x=ddd['lag_0'], y=ddd['lag_1'], z=ddd['lag_2'],
        				mode='lines',
                	#mode='markers',
                    marker=dict(
                        size=4,
                    ),
                    line=dict(
                        color='darkblue',
                        width=0
                    )
                ),
                row=1, col=1)

    fig.append_trace(
                go.Scatter3d(
                    x=dd['lag_0'], y=dd['lag_1'], z=dd['lag_2'],
        				mode='lines',
                	#mode='markers',
                    marker=dict(
                        size=4,
                    ),
                    line=dict(
                        color='darkblue',
                        width=0
                    )
                ),
                row=1, col=2)


    fig.append_trace(
                go.Scatter3d(
                    x=dd_ar['lag_0'], y=dd_ar['lag_1'], z=dd_ar['lag_2'],
        				mode='lines',
                	#mode='markers',
                    marker=dict(
                        size=4,
                    ),
                    line=dict(
                        color='darkblue',
                        width=0
                    )
                ),
                row=1, col=3)

   
    fig.show()



    if plot:
        fig = plt.figure(figsize=(12, 8))
        ax = Axes3D(fig)

        ax.set_xlabel("\n \n  $t$", fontsize=20)
        ax.set_ylabel("\n \n  $t-1$", fontsize=20)
        ax.set_zlabel("\n \n  $t-2$", fontsize=20)
        ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
        ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
        ax.w_zaxis.set_pane_color((0., 0., 0., 0.))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(axis='z', labelsize=20)
        ax.view_init(elev=45, azim=45)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        plt.xticks(np.arange(0, 1.1, 0.2))
        
        if method=='scatter':
            ax.scatter(dd['lag_0'],dd['lag_1'],dd['lag_2'],marker="o")
            if title != None:
                ax.set_title(title, fontsize=30)
        elif method=='line':
            ax.plot(dd_ar['lag_0'],dd_ar['lag_1'],dd_ar['lag_2'],marker= None)
            if title != None:
                ax.set_title(title, fontsize=30,pad=30)
                    
        plt.legend(loc='best', fontsize=20)
        plt.show()
                    


if __name__=='__main__':
    ############

    target = '05e'
    interval = 18
    delay = 7

    p=0
    depth=6
    depth_ar=6

    ############

    ##  load data
    # df_sum1 : Data on day 1
    # df_sum2 : Data on day 2
    df_sum1, df_sum2 = load_data.load_data(interval=interval,p=p)
    df_sum = df_sum1 # choose day

    plot_attractor(df_sum, target, delay, use_wcols=5,tau=1,depth=depth,depth_ar=depth_ar,plot=None,title=None)

    

