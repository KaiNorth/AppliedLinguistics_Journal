# -*- coding: utf-8 -*-
"""
Created on Thu May  6 18:27:04 2021

@author: Yash
"""

import textstat
from task1 import complexity_values_combined
import pandas as pd
from functools import reduce


def getSyllableCount(word):
    """
        A function to count syllables
        :param word: word whose syllables will be counted
        :return: number of syllables in the word
    """
    words = word.split(" ")
    count = 0

    for eachWord in words:
        count += textstat.syllable_count(eachWord)

    return count

def complexity_values_combined_task3():
    """
    A function to get the complexity values for each different syllable count for all datasets 

    Returns
    -------
    DataFrame with the syllable counts and complexity values accordingly

    """
    #get the df complexity combined 
    df,df_groups,df2 = complexity_values_combined()
    #get the complexity values of 
    #keys list for the groups 
    keys_list = list(df_groups.groups.keys())
    #Store word length values in this list
    syllable_count= []
    #get the length value for each token
    for i in range(len(keys_list)):
        syllable_count.append(getSyllableCount(keys_list[i]))
    #create a variable for the tokens as the keys_list 
    tokens = keys_list
    #create a new DataFrame
    df1 = pd.DataFrame()
    df1['tokens'] = tokens 
    df1['syllable_count'] = syllable_count
    unique_syllable_counts = df1['syllable_count'].unique()
    df2_groups = df2.groupby(by='lowercase_tokens')
    current_words = []
    df_list_final = [] 
    current_list=[]    
    syllable_count_list = []
    for i in range(len(unique_syllable_counts)):
        syllable_count_list.append(unique_syllable_counts[i])
        df_temp = df1.where(df1['syllable_count']==unique_syllable_counts[i])
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
    df_syllable_count= pd.DataFrame(df_list_final)
    df_syllable_count['syllable_count'] = syllable_count_list
    df_syllable_count['tokens_similar_length'] = current_list
    df_syllable_count.to_csv('Task3.csv')
    return df_syllable_count