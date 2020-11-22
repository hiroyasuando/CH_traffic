import pandas as pd
import numpy as np
import load_data
from utils import * 

############
target = '05e'
rlist=['05e'] # remove_list
delay = 7
interval = 18
p=0. 
depth=6
depth_ar=6
############

## load data
# df_sum1 : data on day 1
# df_sum2 : data on day2
df_sum1, df_sum2 = load_data.load_data(interval=interval,p=p)


## Prediction within one day (day1)
roads, W, tr_pre, te_pre,_,_,ar_te_pre,_= predict(df_sum1, target=target, remove_list=rlist, delay=delay,depth=depth,depth_ar=depth_ar,plot=True)
n, cols, Fs, Ws, freqs = get_n_Cols_Fs(df_sum1, target, roads, W,depth,interval)
plot_period(df_sum1, cols, Fs, Ws, freqs,interval, n,depth)    # plot power spectrum
print('Train NRMSE : ',tr_pre)
print('Test  NRMSE : ',te_pre)
print('Test(AR)  NRMSE : ',ar_te_pre)

