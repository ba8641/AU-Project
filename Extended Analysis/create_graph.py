# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:13:13 2023

@author: ba8641
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
import sys
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.axes import Axes

data_path = r'/home/ba8641/Extended Analysis'

summary = pd.read_csv(data_path+"/summary.csv").iloc[:,1:]
Var = pd.read_csv(data_path+"/var.csv")

markets = pd.unique(summary['Markets'])
mv = summary.loc[:,['TSS','TCS','TPF']].max().max()
#mv = 1

summary.loc[:,'Production'] = 'N'
summary.loc[:,'AU'] = 'Y'
summary.loc[summary['Markets'].str.strip().str[-1]=='Y','Production'] = 'Y'
summary.loc[~summary['Markets'].str.strip().str[-1].isin(['Y','N']),'AU'] = 'N'
summary.loc[summary['Markets'].str.strip().str[-1]=='Y','Markets'] = summary.loc[summary['Markets'].str.strip().str[-1]=='Y','Markets'].str.strip().str[:-2]
summary.loc[summary['Markets'].str.strip().str[-1]=='N','Markets'] = summary.loc[summary['Markets'].str.strip().str[-1]=='N','Markets'].str.strip().str[:-2]

s = summary.copy().rename(columns={'Min Price Rate':'MPR', 'Markets':'Order'})

model = ols("""TCS ~ C(Market) + C(Order) + C(Commitment) + C(MPR)""", data=s).fit()

sm.stats.anova_lm(model, typ=2)
print(sm.stats.anova_lm(model, typ=2))


s = summary.loc[(summary['Markets'].str.contains('AU')) & (summary['Production']=='N')]
s.loc[:,'AU'] = 'Exists'
s2 = summary.loc[~(summary['Markets'].str.contains('AU'))]
s2.loc[:,'AU'] = 'No AU'
s = pd.concat([s,s2])
model = ols("""TCS ~ C(AU)""", data=s).fit()
print(sm.stats.anova_lm(model, typ=2))
model = ols("""TPF ~ C(AU)""", data=s).fit()
print(sm.stats.anova_lm(model, typ=2))

colors = ['green','red','blue', 'red','blue']
for k in ['TSS', 'TCS', 'TPF']:
    dic = s.boxplot(k, by=['AU','Market'], rot = 90, fontsize=16, figsize = (14,14), patch_artist=True, return_type='dict')
    for patch,color in zip(dic[k]['boxes'],colors):
        patch.set_facecolor(color)
        #[item.set_color('g') for item in bp[key]['boxes']]
    noman = plt.gca()
    noman.set_title(k, fontsize=20)
    noman.set_ylabel("Billion $", fontsize=18)
    noman.set_xlabel("AU Existence, Market", fontsize=18)
    #noman.tick_params(axis='y', labelsize=18)
    plt.suptitle("")
    noman.get_figure().savefig('AU-'+k+'-effect.jpg')
    plt.close()
    
#AU exists
s = summary.loc[(summary['Markets'].str.contains('AU'))]

model = ols("""TCS ~ C(Production)""", data=s).fit()
print(sm.stats.anova_lm(model, typ=2))
model = ols("""TPF ~ C(Production)""", data=s).fit()
print(sm.stats.anova_lm(model, typ=2))

colors = ['green','red','blue','green','red','blue']
for k in ['TSS', 'TCS', 'TPF']:
    dic = s.boxplot(k, by=['Production','Market'], rot = 90, fontsize=16, figsize = (14,14), patch_artist=True, return_type='dict')
    for patch,color in zip(dic[k]['boxes'],colors):
        patch.set_facecolor(color)
        #[item.set_color('g') for item in bp[key]['boxes']]
    noman = plt.gca()
    noman.set_title(k, fontsize=20)
    noman.set_ylabel("Billion $", fontsize=18)
    noman.set_xlabel("AU Production, Market", fontsize=18)
    #noman.tick_params(axis='y', labelsize=18)
    plt.suptitle("")
    noman.get_figure().savefig('Production-'+k+'effect.jpg')
    plt.close()


#Given AU and production
s = summary.copy().rename(columns={'Min Price Rate':'MPR', 'Markets':'Order'})
s = s.loc[(s['Order'].str.contains('AU')) & (s['Production']=='Y')]
summary = s

model = ols("""TCS ~ C(Market)+C(Order)+C(MPR)""", data=s).fit()
print(sm.stats.anova_lm(model, typ=2))
model = ols("""TPF ~ C(Market)+C(Order)+C(MPR)""", data=s).fit()
print(sm.stats.anova_lm(model, typ=2))

#AU Commitment
s = s.loc[(s['Market']=='AU')]
model = ols("""TCS ~ C(Commitment)""", data=s).fit()
print(sm.stats.anova_lm(model, typ=2))
model = ols("""TPF ~ C(Commitment)""", data=s).fit()
print(sm.stats.anova_lm(model, typ=2))
for k in ['TSS', 'TCS', 'TPF']:
    dic = s.boxplot(k, by=['Commitment'], rot = 90, fontsize=16, figsize = (14,14), patch_artist=True, return_type='dict')
    for patch in dic[k]['boxes']:
        patch.set_facecolor('green')
    noman = plt.gca()
    noman.set_title(k, fontsize=20)
    noman.set_ylabel("Billion $", fontsize=18)
    noman.set_xlabel("AU Commitment", fontsize=18)
    #noman.tick_params(axis='y', labelsize=18)
    plt.suptitle("")
    noman.get_figure().savefig('Commitment-'+k+'effect.jpg')
    plt.close()


#Remaining
summary = summary.loc[(summary['Commitment']==0)]
#s = s.loc[(s['MPR'] in [1,2]),:]
#model = ols("""TCS ~ C(Market) + C(Order) + C(MPR) + C(Market)*C(Order) + C(Market)*C(MPR) + C(MPR)*C(Order)""", data=s).fit()
#model = ols("""TCS ~ C(Market) + C(Order) + C(MPR)""", data=s).fit()
#print(sm.stats.anova_lm(model, typ=2))

#comm = [-1,0,.25,.5,.75,1]
colors = ['green','red','blue']
for i in pd.unique(summary['Order']):
    for j in pd.unique(summary['MPR']):
        s = summary.loc[(summary['Order']==i) & (summary['MPR']==j),:]
        if len(s) < 2: continue
        for k in ['TSS','TCS','TPF']:
            s.loc[:,k]=s[k].values/1e9
            dic = s.boxplot(k, by=['Market'], rot = 90, fontsize=16, figsize = (14,14), patch_artist=True, return_type='dict')
            for patch,color in zip(dic[k]['boxes'],colors):
                patch.set_facecolor(color)
                #[item.set_color('g') for item in bp[key]['boxes']]
            noman = plt.gca()
            noman.set_title(k+' when MPR is '+str(j)+' and negotiation order '+str(i), fontsize=20)
            noman.set_ylabel("Billion $", fontsize=18)
            noman.set_xlabel("Market", fontsize=18)
            #noman.tick_params(axis='y', labelsize=18)
            plt.suptitle("")
            noman.get_figure().savefig('Fig_'+k+'_'+i+'_'+str(j)+'.jpg')
            plt.close()

colors = ['green','green','red','red','blue','blue']
for i in pd.unique(summary['Market']):
    for j in pd.unique(summary['MPR']):
        s = summary.loc[(summary['Market']==i) & (summary['MPR']==j),:]
        if len(s) < 2: continue
        for k in ['TSS','TCS','TPF']:
            #s.loc[:,k]=s[k].values/mv
            dic = s.boxplot(k, by=['Order'], rot = 90, fontsize=16, figsize = (14,14), patch_artist=True, return_type='dict')
            for patch,color in zip(dic[k]['boxes'],colors):
                patch.set_facecolor(color)
                #[item.set_color('g') for item in bp[key]['boxes']]
            noman = plt.gca()
            noman.set_title(k+' in '+i+' when MPR is '+str(j)+'00%', fontsize=20)
            noman.set_ylabel("Billion $", fontsize=18)
            noman.set_xlabel("Negotiation Order", fontsize=18)
            #noman.tick_params(axis='y', labelsize=18)
            plt.suptitle("")
            noman.get_figure().savefig('Market/Fig_'+k+'_'+i+'_'+str(j)+'.jpg')
            plt.close()
'''
for i in pd.unique(summary['Market']):
    for j in pd.unique(summary['MPR']):
        for n in range(1,6):
            s = summary.loc[(summary['Market']==i) & (summary['MPR']==j) & (summary['Commitment']<=comm[n]) & (summary['Commitment']>comm[n-1]),:]
            if len(s) < 2: continue
            for k in ['TSS','TCS','TPF']:
                #s.loc[:,k]=s[k].values/mv
                noman = s.boxplot(k, by=['Order'], rot = 90, fontsize=10, figsize = (14,14))
                noman.get_figure().savefig('By Market/Fig_'+k+'_'+i+'_'+str(j)+'.jpg')
                plt.close()
                '''
'''
for i in pd.unique(summary['Market']):
    for j in pd.unique(summary['MPR']):
        s = summary.loc[(summary['Market']==i) & (summary['MPR']==j),:]
        if len(s) < 2: continue
        for k in ['TSS','TCS','TPF']:
            #s.loc[:,k]=s[k].values/mv
            noman = s.boxplot(k, by=['Order'], rot = 90, fontsize=10, figsize = (14,14))
            noman.get_figure().savefig('By Market/Fig_'+k+'_'+i+'_'+str(j)+'.jpg')
            plt.close()
            '''
'''
noman = s.boxplot('TCS', by=['Commitment'], rot = 90, fontsize=10, figsize = (14,14))
noman.get_figure().savefig('Commit-effect.jpg')
plt.close()
'''