import pandas as pd
import numpy as np
import load_data0902
from utils0902 import * 

############
target = '05e'
rlist=['05e']
delay = 7
interval = 18
p=0. 
param=0.3
th=0.5
depth=6
depth_ar=6
############

## データ読み込み
# df_sum1 : 14日のデータ
# df_sum2 : 18日のデータ
df_sum1, df_sum2 = load_data0902.load_data(interval=interval,p=p)


##1日だけで予測(14日)
roads, W, tr_pre, te_pre,_,_,ar_te_pre,_,te_r2,ar_te_r2= predict(df_sum1, target=target, remove_list=rlist, delay=delay,param=param,th=th,depth=depth,depth_ar=depth_ar,plot=True)
n, cols, Fs, Ws, freqs = get_n_Cols_Fs(df_sum1, target, roads, W,depth,interval)
plot_W(target, roads, W,depth)                           # 重み描画
plot_period(df_sum1, cols, Fs, Ws, freqs,interval, n,depth)    # 時系列描画(FFT)
# print('Train NRMSE : ',tr_pre)
print('Test  NRMSE : ',te_pre,' R^2 : ',te_r2)
print('Test(AR)  NRMSE : ',ar_te_pre,' R^2 : ',ar_te_r2)
# print(np.var(W))


#  ### 1日だけで予測(18日)
# roads, W, tr_pre, te_pre,_,_,ar_te_pre,_ = predict(df_sum2, target=target, remove_list=None, delay=delay,param=param,th=th,depth=depth,depth_ar=depth_ar,plot=True)
# # n, cols, Fs, Ws = get_n_Cols_Fs(df_sum2, target, roads, W)
# # plot_W(target, roads, W)                           # 重み描画
# # plot_period(df_sum1, cols, Fs, Ws, interval, n)    # 時系列描画(FFT)
# #print('self Train NRMSE : ',tr_pre)
# print('self Test  NRMSE : ',te_pre)
# print('Test(AR)  NRMSE : ',ar_te_pre)
# print(np.var(W))

###------------------------------------------------------------------------
## 各日にちでWを計算
# 14日データで学習した重み
# W1 = calc_W(df_sum1, target=target, remove_list=['05e'], delay=delay,param=param,depth=depth)
# W1 = calc_W_rob(df_sum1, target=target, remove_list=rlist, delay=delay,param=param,depth=depth)
# # 18日データで学習した重み
# W2 = calc_W(df_sum2, target=target, remove_list=None, delay=delay,param=param,depth=depth)
# print(W1.shape)
# print(W2.shape)
# predict_by_W_rob(df_sum1, W1, target, delay,depth,param,plot=True,remove_list=rlist)

# 14の重みで18を予測する
# print('cross 14->18  NRMSE : ')
# predict_by_W(df_sum2, W1, target, delay,depth,param,plot=True)
# # 18の重みで14を予測する
# print('cross 18->14  NRMSE : ')
# predict_by_W(df_sum1, W2, target, delay,depth,param,plot=True)

## リアルタイム予測
# 14日の後ろに18日をつける
# df_sum = pd.concat([df_sum1, df_sum2]).reset_index(drop=True)
# real_time_predict(df_sum, target=target, delay=delay,param=param,depth=depth)
# 14日でリアルタイム予測
# real_time_predict(df_sum1, target=target, delay=delay,param=param,depth=depth)
# 18日でリアルタイム予測
# real_time_predict(df_sum2, target=target, delay=delay,param=param,depth=depth)
