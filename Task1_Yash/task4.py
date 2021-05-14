# -*- coding: utf-8 -*-
"""
Created on Sun May  9 15:02:04 2021

@author: Yash
"""

from task1 import *

def complexity_values_combined_task4():
    """
    Get the complexity values of each token with frequency counts in each

    Returns
    -------
    DataFrames with 2 complexity counts 

    """
    #Get the complexity values
    df_CompLex = complexity_ratings_CompLex()
    df_WCL = complexity_ratings_WCL()
    df_SemEval2016 = complexity_ratings_SemEval2016()
    df_BEA2018_1,df_BEA2018_2,df_BEA2018_3 = complexity_ratings_BEA2018()
    #get the combined complexity values
    df_combined = complexity_values()
    tokens_CompLex = df_CompLex['token']
    tokens_WCL = df_WCL['token']
    tokens_SemEval2016 = df_SemEval2016['token']
    tokens_BEA2018_1 = df_BEA2018_1['token']
    tokens_BEA2018_2 = df_BEA2018_2['token']
    tokens_BEA2018_3 = df_BEA2018_3['token']
    #get the frequency counts
    frequency1 = 0
    frequency2 = 0
    frequency3 = 0
    frequency4 = 0
    frequency5 = 0
    frequency6 = 0
    frequency7 = 0
    #get the sum_complexity
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    sum6 = 0
    sum7 = 0
    sum8 = 0
    sum9 = 0
    tokens_compare = []
    #get the tokens
    tokens1 = []
    tokens2 = []
    tokens3 = []
    tokens4 = []
    tokens5 = []
    tokens6 = []
    #get the averages
    avg1 = []
    avg2 = []
    avg3 = []
    avg4 = []
    avg5 = []
    avg6 = []
    avg7 = []
    avg8 = []
    avg9 = []
    frequencySeries=[]
    for i in range(len(df_combined)):
        tokens_compare.append(str(df_combined['token'][i]).lower())
    df_combined['lowercase_tokens'] = tokens_compare
    for j in range(len(df_combined)):
        for i in range(len(tokens_CompLex)):
            if (df_combined['lowercase_tokens'][j] == tokens_CompLex[i]):
                frequency1 += 1
                sum1 += df_combined['CompLex_complexity'][j]
        if (frequency1!=0):
            avg1.append(sum1/frequency1)
            tokens1.append(df_combined['lowercase_tokens'][i])
        frequencySeries.append(frequency1)
        for k in range(len(tokens_WCL)):    
            if (df_combined['lowercase_tokens'][j] == tokens_WCL[i]):
                frequency2 += 1
                sum2 += df_combined['WCL_complexity'][j]
        if (frequency2!=0):
            avg2.append(sum2/frequency2)
            tokens2.append(df_combined['lowercase_tokens'][j])
        frequencySeries.append(frequency2)
        for l in range(len(tokens_SemEval2016)): 
            if (df_combined['lowercase_tokens'][j] == tokens_SemEval2016[i]):
                frequency3 += 1
                sum3 += df_combined['SemEval2016_complexity'][j]
        if (frequency3!=0):
            avg3.append(sum3/frequency3)
            tokens3.append(df_combined['lowercase_tokens'][j])
        frequencySeries.append(frequency3)
        for m in range(len(tokens_BEA2018_1)):    
            if (df_combined['lowercase_tokens'][j] == tokens_BEA2018_1[i]):
                frequency4 += 1
                sum4 += df_combined['BEA2018_News_binaryComplexity'][j]
                sum5 += df_combined['BEA2018_News_probabilisticComplexity'][j]
        if (frequency4!=0):        
            avg4.append(sum4/frequency4)
            avg5.append(sum5/frequency4)
            tokens4.append(df_combined['lowercase_tokens'][j])
        frequencySeries.append(frequency4)
        for n in range(len(tokens_BEA2018_2)):
            if (df_combined['lowercase_tokens'][j] == tokens_BEA2018_2[i]):
                frequency5 += 1
                sum6 += df_combined['BEA2018_WikiNews_binaryComplexity'][j]
                sum7 += df_combined['BEA2018_WikiNews_probabilisticComplexity'][j]
        if (frequency5!=0):      
            avg6.append(sum6/frequency5)
            avg7.append(sum7/frequency5)
            tokens5.append(df_combined['lowercase_tokens'][j])
        frequencySeries.append(frequency5)
        for o in range(len(tokens_BEA2018_3)):
            if (df_combined['lowercase_tokens'][j] == tokens_BEA2018_3[i]):
                frequency6 += 1
                sum8 += df_combined['BEA2018_Wikipedia_binaryComplexity'][j]
                sum9 += df_combined['BEA2018_Wikipedia_probabilisticComplexity'][j]
        if (frequency6!=0):  
            avg8.append(sum7/frequency6)
            avg9.append(sum8/frequency6)
            tokens6.append(df_combined['lowercase_tokens'][j])
        frequencySeries.append(frequency6)
    df_combined['frequencySeries'] = frequencySeries
    return df_combined