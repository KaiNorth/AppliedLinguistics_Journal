# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:12:06 2021

@author: Yash
"""

import os
import pandas as pd 
import csv
  
    
"""
Get the complexity values for the CompLex database
"""    
def complexity_ratings_CompLex():
    #getting current directory
    current_path = os.getcwd()
    #split current_path by "\\"
    split_val = current_path.split('\\')
    #check if base_dir is in correct_path ,i.e., where the AppliedLinguistic_Journal folder (root of git repo is) 
    if (split_val[-1]=="AppliedLinguistics_Journal"):
        #join the dir_path by /
        dir_path = "/".join(split_val)
        #get the path to the CompLex dataset
        dataset_CompLex_path =  dir_path + '/Applied Linguistics Journal/Datasets/CompLex/CompLex-master/lcp_multi_train.tsv'
        #get the file contents in a dataFrame
        df_CompLex = pd.read_csv(dataset_CompLex_path,sep='\t')
    elif (split_val[-1]!="AppliedLinguistics_Journal"): 
        #Enter the path to the cloned git repository
        base_dir = input('Enter the path to the git repository (Main folder named: AppliedLinguistics_Journal) with forward slashes: ')
        #Find the file path
        dataset_CompLex_path = base_dir + '/Applied Linguistics Journal/Datasets/CompLex/CompLex-master/lcp_multi_train.tsv'
        #get the file contents in a dataFrame
        df_CompLex = pd.read_csv(dataset_CompLex_path,sep='\t')
    return df_CompLex
    
"""
Get the complexity values for the WCL
"""      
def complexity_ratings_WCL():
    #getting current directory
    current_path = os.getcwd()
    #split current_path by "\\"
    split_val = current_path.split('\\')
    #check if base_dir is in correct_path ,i.e., where the AppliedLinguistic_Journal folder (root of git repo is) 
    if (split_val[-1]=="AppliedLinguistics_Journal"):
        #join the dir_path by /
        dir_path = "/".join(split_val)
        #get the path to the CompLex dataset
        dataset_WCL_path =  dir_path + '/Applied Linguistics Journal/Datasets/WCL/word_complexity_lexicon/lexicon.tsv'
        #get the file contents in a dataFrame
        df_WCL = pd.read_csv(dataset_WCL_path,sep='\t',header=None)
    elif (split_val[-1]!="AppliedLinguistics_Journal"): 
        #Enter the path to the cloned git repository
        base_dir = input('Enter the path to the git repository (Main folder named: AppliedLinguistics_Journal) with forward slashes: ')
        #Find the file path
        dataset_WCL_path = base_dir + '/Applied Linguistics Journal/Datasets/WCL/word_complexity_lexicon/lexicon.tsv'
        #get the file contents in a dataFrame
        df_WCL = pd.read_csv(dataset_WCL_path,sep='\t',header=None)
    df_WCL.columns = ['token','complexity']     
    return df_WCL  
        
"""
Get the complexity values for the SemEval2016
"""      
def complexity_ratings_SemEval2016():
    #getting current directory
    current_path = os.getcwd()
    #split current_path by "\\"
    split_val = current_path.split('\\')
    #check if base_dir is in correct_path ,i.e., where the AppliedLinguistic_Journal folder (root of git repo is) 
    if (split_val[-1]=="AppliedLinguistics_Journal"):
        #join the dir_path by /
        dir_path = "/".join(split_val)
        #get the path to the CompLex dataset
        dataset_SemEval2016_path =  dir_path + '/Applied Linguistics Journal/Datasets/SemEval-2016/cwi_testing_annotated.txt'
        #get the file contents in a dataFrame
        df_SemEval2016 = pd.read_csv(dataset_SemEval2016_path,sep='\t',header=None)
    elif (split_val[-1]!="AppliedLinguistics_Journal"): 
        #Enter the path to the cloned git repository
        base_dir = input('Enter the path to the git repository (Main folder named: AppliedLinguistics_Journal) with forward slashes: ')
        #Find the file path
        dataset_SemEval2016_path = base_dir + '/Applied Linguistics Journal/Datasets/SemEval-2016/cwi_testing_annotated.txt'
        #get the file contents in a dataFrame
        df_SemEval2016 = pd.read_csv(dataset_SemEval2016_path,sep='\t',header=None)
    df_SemEval2016.columns = ['sentences','token','placement','Complexity']     
    return df_SemEval2016        

        
"""
Get the complexity values for the BEA-2018
"""      
def complexity_ratings_BEA2018():
    #getting current directory
    current_path = os.getcwd()
    #split current_path by "\\"
    split_val = current_path.split('\\')
    #check if base_dir is in correct_path ,i.e., where the AppliedLinguistic_Journal folder (root of git repo is) 
    if (split_val[-1]=="AppliedLinguistics_Journal"):
        #join the dir_path by /
        dir_path = "/".join(split_val)
        #get the path to the CompLex dataset
        dataset_BEA2018_path1 =  dir_path + '/Applied Linguistics Journal/Datasets/BEA-2018/traindevset/english/News_Train.tsv'
        dataset_BEA2018_path2 =  dir_path + '/Applied Linguistics Journal/Datasets/BEA-2018/traindevset/english/WikiNews_Train.tsv'
        dataset_BEA2018_path3 =  dir_path + '/Applied Linguistics Journal/Datasets/BEA-2018/traindevset/english/Wikipedia_Train.tsv'
        #get the file contents in a dataFrame
        df_BEA2018_1 = pd.read_csv(dataset_BEA2018_path1,sep='\t',header=None)
        df_BEA2018_2 = pd.read_csv(dataset_BEA2018_path2,sep='\t',header=None)
        df_BEA2018_3 = pd.read_csv(dataset_BEA2018_path3,sep='\t',header=None)
    elif (split_val[-1]!="AppliedLinguistics_Journal"): 
        #Enter the path to the cloned git repository
        base_dir = input('Enter the path to the git repository (Main folder named: AppliedLinguistics_Journal) with forward slashes: ')
        #Find the file path to each of the tsv - News
        dataset_BEA2018_path1 = base_dir + '/Applied Linguistics Journal/Datasets/BEA-2018/News_Train.tsv'
        dataset_BEA2018_path2 =  dir_path + '/Applied Linguistics Journal/Datasets/BEA-2018/WikiNews_Train.tsv'
        dataset_BEA2018_path3 =  dir_path + '/Applied Linguistics Journal/Datasets/BEA-2018/Wikipedia_Train.tsv'
        #get the file contents in a dataFrame
        df_BEA2018_1 = pd.read_csv(dataset_BEA2018_path1,sep='\t',header=None)
        df_BEA2018_2 = pd.read_csv(dataset_BEA2018_path2,sep='\t',header=None)
        df_BEA2018_3 = pd.read_csv(dataset_BEA2018_path3,sep='\t',header=None)
    df_BEA2018_1.columns = ['HIT_ID','sentence','start','end','token','NA_See_1','NA_See_2','NA_mark_1','NA_mark_2','binaryComplexity',
                            'probabilisticComplexity']  
    df_BEA2018_2.columns = ['HIT_ID','sentence','start','end','token','NA_See_1','NA_See_2','NA_mark_1','NA_mark_2','binaryComplexity',
                            'probabilisticComplexity'] 
    df_BEA2018_3.columns = ['HIT_ID','sentence','start','end','token','NA_See_1','NA_See_2','NA_mark_1','NA_mark_2','binaryComplexity',
                            'probabilisticComplexity'] 
    return df_BEA2018_1,df_BEA2018_2,df_BEA2018_3        

"""
Get Complexity values for all 4 datasets
"""
def complexity_values():
    df_CompLex = complexity_ratings_CompLex()
    df_WCL = complexity_ratings_WCL()
    df_SemEval2016 = complexity_ratings_SemEval2016()
    df_BEA2018_1,df_BEA2018_2,df_BEA2018_3 = complexity_ratings_BEA2018()
    df_final = pd.DataFrame()
    df_final['token'] = df_CompLex['token']
    df_final['CompLex_complexity'] = df_CompLex['complexity']
    df_final2 = pd.DataFrame()
    df_final2['token'] = df_WCL['token']
    df_final2['WCL_complexity'] = df_WCL['complexity']
    df_final3 = pd.DataFrame()
    df_final3['token'] = df_SemEval2016['token']
    df_final3['SemEval2016_complexity'] = df_WCL['complexity']
    df_final4 = pd.DataFrame()
    df_final4['token'] = df_BEA2018_1['token']
    df_final4['BEA2018_News_binaryComplexity'] = df_BEA2018_1['binaryComplexity']
    df_final4['BEA2018_News_probabilisticComplexity'] = df_BEA2018_1['probabilisticComplexity']
    df_final5 = pd.DataFrame()
    df_final5['token'] = df_BEA2018_2['token']
    df_final5['BEA2018_WikiNews_binaryComplexity'] = df_BEA2018_2['binaryComplexity']
    df_final5['BEA2018_WikiNews_probabilisticComplexity'] = df_BEA2018_2['probabilisticComplexity']
    df_final6 = pd.DataFrame()
    df_final6['token'] = df_BEA2018_3['token']
    df_final6['BEA2018_Wikipedia_binaryComplexity'] = df_BEA2018_3['binaryComplexity']
    df_final6['BEA2018_Wikipedia_probabilisticComplexity'] = df_BEA2018_3['probabilisticComplexity']
    df_last = pd.concat([df_final,df_final2,df_final3,df_final4,df_final5,df_final6],axis=0)
    df_last.reset_index(inplace=True)
    df_last.drop(labels='index',axis=1,inplace=True)
    return df_last




