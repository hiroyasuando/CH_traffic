import pandas as pd
import numpy as np
import load_data
from utils import * 

############
target = '05e'
rlist=['05e'] # remove_list
delay = 7
interval = 17
p=0. 
depth=2
depth_ar=6
############

## load data
# df_sum1 : data on day 1
# df_sum2 : data on day2
df_sum1, df_sum2 = load_data.load_data(interval=interval,p=p)



# Learn W_out on day 1
W1 = calc_W(df_sum1, target=target, remove_list=None, delay=delay,depth=depth)

# prediction from W of day 1 to day 2
print('cross 14->18  NRMSE : ')
predict_by_W(df_sum2, W1, target, delay,depth,plot=True)
