# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:43:59 2021

@author: Yash
"""

from nltk import pos_tag 
from task1 import *
import pandas as pd
from functools import reduce

def complexity_values_combined_task6():
    #get the df complexity combined 
    df,df_groups,df2 = complexity_values_combined()
    #get the complexity values of 
    #keys list for the groups 
    keys_list = list(df_groups.groups.keys())
    #get the length value for each token
    ed_keywords = []
    value = False
    for i in range(len(keys_list)):
        words = keys_list[i].split(' ')
        if ((len(words)==1)):
            if (pos_tag(words[0])[1] =='VBD' or pos_tag(words[0])[1] == 'VBG'):
                if (len(words[0])>2):
                    if (words[0][:-2]=='-ed'):
                        pass
                    else: 
                        ed_keywords.append(keys_list[i])
        elif (len(words)!=1):
            for j in range(len(words)):
                if (pos_tag(words[j])[1] =='VBD' or pos_tag(words[j])[1] == 'VBG'):
                    if (words[j][:-2]=='-ed'):
                        pass
                    else: 
                        value=True
            if (value==True):
                ed_keywords.append(keys_list[i])
    #create a variable for the tokens as the verbs with -"ed"
    tokens = ed_keywords
    #create a new DataFrame
    df1 = pd.DataFrame()
    df1['tokens'] = ed_keywords
    df2_groups = df2.groupby(by='lowercase_tokens')
    df_list_final = []
    df2_list_temp = []
    df_word_length = []
    for i in range(len(ed_keywords)):
        #df_temp = df1.where(df1['tokens']==ed_keywords[i])
        #3df_temp.dropna(inplace=True)
        #current_words = df_temp['tokens'].to_list()
        #current_list.append(current_words)
        #for j in range(len(current_words)):
            #if (len(current_words[j])==1):
                #pass
            #else: 
        df2_list_temp.append(df2.iloc[df2_groups.groups[ed_keywords[i]].values])
        df2_list_temp_new = []
        for k in range(len(df2_list_temp)):
            df2_list_temp_new.append(df2_list_temp[k].drop(labels=['token','lowercase_tokens'],axis=1))
        df_list_final.append(reduce(lambda a, b: a.add(b, fill_value=0), df2_list_temp_new).mean())
    df_word_length = pd.DataFrame(df_list_final)
    df_word_length['tokens'] = tokens
    df_word_length.to_csv('Task6.csv')
    return df_word_length