# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:50:14 2021

@author: Yash
"""

from task1 import complexity_values_combined
from task1 import complexity_values
import pandas as pd
from functools import reduce

"""
The complexity values combined are used for Task 2 
"""
def complexity_values_combined_task2():
    #get the df complexity combined 
    df,df_groups,df2 = complexity_values_combined()
    #get the complexity values of 
    #keys list for the groups 
    keys_list = list(df_groups.groups.keys())
    #Store word length values in this list
    word_length = []
    #get the length value for each token
    for i in range(len(keys_list)):
        word_length.append(len(keys_list[i]))
    #create a variable for the tokens as the keys_list 
    tokens = keys_list
    #create a new DataFrame
    df1 = pd.DataFrame()
    df1['tokens'] = tokens 
    df1['word_length'] = word_length
    unique_word_lengths = df1['word_length'].unique()
    df2_groups = df2.groupby(by='lowercase_tokens')
    current_words = []
    df_list_final = [] 
    current_list=[]    
    word_length_list = []
    for i in range(len(unique_word_lengths)):
        if (unique_word_lengths[i]==1):
            pass
        else: 
            word_length_list.append(unique_word_lengths[i])
            df_temp = df1.where(df1['word_length']==unique_word_lengths[i])
            df_temp.dropna(inplace=True)
            current_words = df_temp['tokens'].to_list()
            df2_list_temp = []
            current_list.append(current_words)
            for j in range(len(current_words)):
                if (len(current_words[j])==1):
                    pass
                else: 
                    df2_list_temp.append(df2.iloc[df2_groups.groups[current_words[j]].values])
            df2_list_temp_new = []
            for k in range(len(df2_list_temp)):
                df2_list_temp_new.append(df2_list_temp[k].drop(labels=['token','lowercase_tokens'],axis=1))
            df_list_final.append(reduce(lambda a, b: a.add(b, fill_value=0), df2_list_temp_new).mean())
    df_word_length = pd.DataFrame(df_list_final)
    df_word_length['word_length'] = word_length_list
    df_word_length['tokens_similar_length'] = current_list
    df_word_length.to_csv('Task2.csv')
    return df_word_length     
            