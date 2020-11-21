import pandas as pd
import numpy as np
import load_data0902
from utils0902 import * 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#3次元プロットするためのモジュール
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import warnings
warnings.filterwarnings('ignore')


def plot_attractor(df_sum, target, delay,use_wcols=3,tau=1,param=1,depth=1,depth_ar=5,method='line',title='11e',plot=False):
    '''
    use_wcols : 重みが大きい地点を何地点を時系列として結合するか
    '''

    ## targetの時系列
    # ddd = df_sum[[target]].rename(columns={target:'lag_0'})
    # for i in range(1,5):
    #     ddd[f'lag_{i}'] = df_sum[target].shift(tau*i)
    # ddd = ddd.dropna()

    roads, w, _, _,pred, pred_ar,_,ts = predict(df_sum, target=target, remove_list=None, delay=delay,lam=0.05,param=param,depth=depth,depth_ar=depth_ar,plot=False)
    df_W = pd.DataFrame({'X':roads, 'W':w, 'W_abs': np.abs(w)})
    

    ddd = pd.DataFrame()
    ddd['lag_0'] = ts
    for i in range(1,5):
        ddd[f'lag_{i}'] = ddd['lag_0'].shift(tau*i)
    ddd = ddd.dropna()

    
    # df_X = df_sum.drop(target,axis=1)
    # for col in df_X.columns:
    #     df_X[col] = df_X[col] / df_X[col].max()
    #     for i in range(1,depth):
    #         df_X[f'{col}lag_{i}'] = df_X[col].shift(i)
    # df_X = df_X.dropna()
    # df_X['bias'] = 1
    

    dd = pd.DataFrame()
    # for x, w in zip(df_W.sort_values('W_abs', ascending=False).iloc[:use_wcols, 0], df_W.sort_values('W_abs', ascending=False).iloc[:use_wcols, 1]):
    #     dd[x] = (df_X[x]/df_X[x].max()) * w
    # dd['lag_0'] = dd.sum(axis=1)
    dd['lag_0'] = pred
    for i in range(1,5):
        dd[f'lag_{i}'] = dd['lag_0'].shift(tau*i)
    dd = dd.dropna()

    # dd_r = pd.DataFrame()
    # for x, w in zip(df_W['X'].sample(n=use_wcols), df_W.sort_values('W_abs', ascending=False).iloc[:use_wcols, 1]):
    #     dd_r[x] = (df_X[x]/ df_X[x].max()) * w
    # dd_r['lag_0'] = dd_r.sum(axis=1)
    # for i in range(1,5):
    #     dd_r[f'lag_{i}'] = dd_r['lag_0'].shift(tau*i)
    # dd_r = dd_r.dropna()

    dd_ar = pd.DataFrame()
    dd_ar['lag_0'] = pred_ar
    for i in range(1,5):
        dd_ar[f'lag_{i}'] = dd_ar['lag_0'].shift(tau*i)
    dd_ar = dd_ar.dropna()
    

    
    # dd_m = pd.DataFrame()
    # for x, w in zip(df_W.sort_values('W_abs', ascending=True).iloc[:use_wcols, 0], df_W.sort_values('W_abs', ascending=True).iloc[:use_wcols, 1]):
    #     dd_m[x] = (df_sum[x]/ df_sum[x].max()) * w
    # dd_m['lag_0'] = dd.sum(axis=1)
    # for i in range(1,5):
    #     dd_m[f'lag_{i}'] = dd_m['lag_0'].shift(tau*i)
    # dd_m = dd_m.dropna()

    ##出力
    dd.to_csv("./wassersteinCC/reconstruct.csv",columns=['lag_0','lag_1','lag_2'],header=False,index=False,sep='\t')
    ddd.to_csv("./wassersteinCC/target.csv",columns=['lag_0','lag_1','lag_2'],header=False,index=False,sep='\t')
    # dd_r.to_csv("./wassersteinCC/random.csv",columns=['lag_0','lag_1','lag_2'],header=False,index=False,sep='\t')
    dd_ar.to_csv("./wassersteinCC/ar.csv",columns=['lag_0','lag_1','lag_2'],header=False,index=False,sep='\t')
   
    ### 描画
    plt.figure(figsize=(24,24))
    fig = make_subplots(rows=1, cols=3,
                    # specs=[[{'is_3d': True}, {'is_3d': True}],[{'is_3d': True},{'is_3d': True}]],
                    specs=[[{'is_3d': True}, {'is_3d': True},{'is_3d': True}]],
                    print_grid=False,
                   subplot_titles=(target, 'Reservoir','AR'))
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

    # fig.append_trace(
    #             go.Scatter3d(
    #                 x=dd_r['lag_0'], y=dd_r['lag_1'], z=dd_r['lag_2'],
    #     				mode='lines',
    #             	#mode='markers',
    #                 marker=dict(
    #                     size=4,
    #                 ),
    #                 line=dict(
    #                     color='darkblue',
    #                     width=0
    #                 )
    #             ),
    #             row=2, col=1)

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
        #グラフの枠を作っていく
        fig = plt.figure(figsize=(12, 8))
        ax = Axes3D(fig)

        #軸にラベルを付けたいときは書く
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
            # ax.scatter(array[:-2],array[1:-1],array[2:],marker="o")
            ax.scatter(dd['lag_0'],dd['lag_1'],dd['lag_2'],marker="o")
            if title != None:
                ax.set_title(title, fontsize=30)
        elif method=='line':
            # ax.plot(array[:-2],array[1:-1],array[2:],marker= None)
            ax.plot(dd_ar['lag_0'],dd_ar['lag_1'],dd_ar['lag_2'],marker= None)
            if title != None:
                ax.set_title(title, fontsize=30,pad=30)
                    
        #plt.legend(loc='best', fontsize=20)
        # plt.show()
        filename = "attar5e_dp"+ str(depth) +".eps"
        fig.savefig(filename, bbox_inches="tight")
                    


if __name__=='__main__':
    ############
    # target = '10e'
    # delay = 6
    # interval = 23

    # target = '11w'
    # interval = 15
    # delay = 1

    target = '05e'
    interval = 18
    delay = 7

    p=0
    param=0.1
    depth=6
    depth_ar=6

    ############

    ## データ読み込み
    # df_sum1 : 14日のデータ
    # df_sum2 : 18日のデータ
    df_sum1, df_sum2 = load_data0902.load_data(interval=interval,p=p)
    df_sum = df_sum1 # 使う日にち

    plot_attractor(df_sum, target, delay, use_wcols=5,tau=1,param=param,depth=depth,depth_ar=depth_ar,plot=True,title='AR')

    

