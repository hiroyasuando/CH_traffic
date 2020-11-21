import pandas as pd
import numpy as np
import os
import re

def create_data(df, file_name, root_dir,date):
    file_path = os.path.join(root_dir, file_name)
    point = file_name.split('.')[0]
    
    # read date
    df_part = pd.read_csv(file_path,header=None).rename(columns={0:'image',1:'date'})
    df_part['date1'] = date
    df_part['date'] = df_part['date1'] + df_part['date']
    df_part.drop('date1',axis=1,inplace=True)
    df_part['date'] = pd.to_datetime(df_part['date'])
    df_part = df_part.sort_values(by='date').reset_index(drop=True)
    
    # add car flag
    date_map = df_part.date.value_counts()
    df_part[point] = df_part.date.map(date_map)
    df_part = df_part.drop_duplicates(subset='date')
    
    df = pd.merge(df, df_part[['date',point]],on='date',how='left')
    df = df.fillna(0)
    
    return df
def sum_df(df,interval):
    in_rows = []
    df_sum = pd.DataFrame()
    for i in range(interval-1,len(df),interval):
        in_rows.append(i)

    in_col = []
    for col in df.iloc[:,1:].columns:
        df_sum[col] = df[col].rolling(interval).sum()
        in_col.append(col)
    df_sum['bias'] = 1
    in_col.append('bias')
    
    df_sum = df_sum.loc[in_rows, in_col].reset_index(drop=True)
    
    return df_sum

def load_data(interval,p):
    file_list1 = os.listdir('./result')
    file_list2 = os.listdir('./result2')
    
    # 交差点以外の部分は排除
    # remove_list = ['01e.csv','01w.csv','01s.csv','01n.csv','02e.csv', '02w.csv','05n.csv','05s.csv',
    #                '06n.csv','06s.csv','07n.csv','07s.csv', '.DS_Store','08n.csv','08s.csv','12n.csv','12s.csv',
    #                '13n.csv','13s.csv','14n.csv','14s.csv','20s.csv','20n.csv','21n.csv','21s.csv','11w.csv','11n.csv','11s.csv']
    remove_list = [ '.DS_Store','05w.csv','05n.csv','05s.csv']

    for rm_file in remove_list:
        try:
            file_list1.remove(rm_file)
        except:
            pass
        try:
            file_list2.remove(rm_file)
        except:
            pass 

    file_list1.sort()
    file_list2.sort()

    # data create
    df1 = pd.DataFrame()
    df1['date'] = pd.date_range(start='2/14/2020 10:50:00', end='2/14/2020 11:49:59', freq='s')
    for file_name in file_list1:
        df1 = create_data(df1, file_name, root_dir='./result', date='2020/02/14 ')

    # data create
    df2 = pd.DataFrame()
    df2['date'] = pd.date_range(start='2/18/2020 07:40:00', end='2/18/2020 08:30:59', freq='s')
    for file_name in file_list2:
        df2 = create_data(df2, file_name,root_dir='./result2',date='2020/02/18 ')


    if p>0:
        for i in range(1, len(df1.columns)):   
            mask = np.random.choice(np.arange(len(df1)), int(len(df1)*p))
            df1.iloc[mask, i] = 0
            
        for i in range(1, len(df2.columns)):   
            mask = np.random.choice(np.arange(len(df2)), int(len(df2)*p))
            df2.iloc[mask, i] = 0
        
    # df_sum1 = sum_df(df1, interval)
    # df_sum2 = sum_df(df2, interval)

    # return df_sum1, df_sum2

    return df1, df2

    
