# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:15:20 2021

@author: Yash
"""

from task4 import *

def complexity_values_combined_task4_2():
    """
    Part 2 for getting averages for the complexity values for each frequency

    Returns
    -------
    DataFrame with average scores for each frequency

    """
    df,df_groups,df2 = complexity_values_combined()
    df1 = complexity_values_combined_task4()
    unique_frequency = df1['frequency'].unique()
    df2_groups = df2.groupby(by='lowercase_tokens')
    current_words = []
    df_list_final = [] 
    current_list=[]    
    frequency_list = []
    for i in range(len(unique_frequency)):
        frequency_list.append(unique_frequency[i])
        df_temp = df1.where(df1['frequency']==unique_frequency[i])
        df_temp.dropna(inplace=True)
        current_words = df_temp['lowercase_tokens'].to_list()
        df2_list_temp = []
        current_list.append(current_words)
        for j in range(len(current_words)):
            df2_list_temp.append(df2.iloc[df2_groups.groups[current_words[j]].values].mean())
    df_frequency= pd.DataFrame(df_list_final)
    df_frequency['frequency'] = frequency_list
    df_frequency['tokens_similar_length'] = current_list
    df_frequency.to_csv('Task4_2.csv')
    return df_frequency