import pandas as pd
import numpy as np
import load_data
from utils import * 

############
target = '05e'
rlist=['05e','09w','09s','08n'] # remove_list
delay = 7
interval = 18
p=0. 
depth=6
depth_ar=6
############

## load data
# df_sum1 : data on day 1
# df_sum2 : data on day 2
df_sum1, df_sum2 = load_data.load_data(interval=interval,p=p)



### Learn W_out after remove

roads, W, tr_pre, te_pre,_,_,ar_te_pre,_= predict(df_sum1, target=target, remove_list=rlist, delay=delay,depth=depth,depth_ar=depth_ar,plot=True)
print('Test  NRMSE : ',te_pre)

