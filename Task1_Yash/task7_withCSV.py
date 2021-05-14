# -*- coding: utf-8 -*-
"""
Created on Thu May 13 22:06:34 2021

@author: Yash
"""

from task1 import * 
import pandas as pd
from functools import reduce

def complexity_values_combined_task7_2():
    """
    Get the complexity values with frequencies by comparing them with the csv file list 
    
    Reference: https://speakspeak.com/resources/vocabulary-general-english/english-irregular-verbs

    Returns
    -------
    DataFrame with complexity values and frequencies for each dataset

    """
    df, df_groups, df2 = complexity_values_combined()
    #get the tokens
    lowercase_tokens = df['lowercase_tokens']
    #initializing some arrays
    current_list = []
    df2_list_temp_new = []
    df_list_final = []
    frequencies = []
    df = pd.read_csv('IrregularVerb.csv',header=None)
    verbs_list = df['Irregular Verb']
    #get the frequency of tokens 
    for i in range(len(verbs_list)):
        df1 = df.where[df['lowercase_tokens']==verbs_list[i]]
        df1 = df1.dropna()
        current_words = df1['lowercase_tokens'].to_list()
        current_list.append(current_words)
        df2_list_temp = []
        for j in range(len(current_words)):
            df2_list_temp.append(df2.iloc[df_groups.groups[current_words[j]].values])
        df2_list_temp_new = []
        for k in range(len(current_words)):
            df2_list_temp_new.append(df2_list_temp[k].drop(labels=['token','lowercase_tokens'],axis=1))
        df_list_final.append(reduce(lambda a, b: a.add(b, fill_value=0), df2_list_temp_new).mean())
        frequencies.append(len(reduce(lambda a, b: a.add(b, fill_value=0), df2_list_temp_new)))
    df_frequency = pd.DataFrame(df_list_final)
    df_frequency['frequencies'] = frequencies
    df_frequency.to_csv('Task4_1.csv')
    return df_frequency