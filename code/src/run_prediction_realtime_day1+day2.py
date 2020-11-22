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
depth=1
depth_ar=6
############

## load data
# df_sum1 : data on day 1
# df_sum2 : data on day2
df_sum1, df_sum2 = load_data.load_data(interval=interval,p=p)


## real time prediction
# Day 1 + Day 2
df_sum = pd.concat([df_sum1, df_sum2]).reset_index(drop=True)
real_time_predict(df_sum, target=target, delay=delay,depth=depth)

# Day 1
# real_time_predict(df_sum1, target=target, delay=delay,depth=depth)

# Day 2
# real_time_predict(df_sum2, target=target, delay=delay,depth=depth)
